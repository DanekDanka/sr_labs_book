"""
Оценка качества системы верификации диктора
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import soundfile as sf

# Добавляем путь к общим модулям
sys.path.append(os.path.realpath('..'))

from common import test_dataset_loader, extract_features, compute_scores_cosine, get_eer
from lab4.LoadModel import load_model_from_lab3
from lab4.ResNetFromLab3 import MainModel
from protocol_generator import load_protocol


def extract_embeddings_for_protocol(
    model,
    protocol: list,
    dataset_base_path: str,
    eval_frames: int = 200,  # ~10 секунд при hop_length=160
    num_eval: int = 5,
    batch_size: int = 1,
    device: str = 'cuda'
):
    """
    Извлекает эмбеддинги для всех файлов в протоколе
    
    Args:
        model: обученная модель
        protocol: список (label, enroll_path, test_path, enroll_year, test_year)
        dataset_base_path: базовый путь к датасету
        eval_frames: количество фреймов для оценки (~10 секунд)
        num_eval: количество сегментов для усреднения
        batch_size: размер батча
        device: устройство (cuda/cpu)
    
    Returns:
        Словарь эмбеддингов: {file_path: embedding_tensor}
    """
    # Собираем все уникальные файлы
    all_files_set = set()
    for label, enroll_path, test_path, _, _ in protocol:
        all_files_set.add(enroll_path)
        all_files_set.add(test_path)
    
    all_files = sorted(list(all_files_set))
    
    # Нормализуем пути к файлам
    file_list = []
    file_path_mapping = {}  # Маппинг оригинального пути к нормализованному
    
    for file_path in all_files:
        # Если путь абсолютный, используем его как есть
        if os.path.isabs(file_path):
            normalized_path = file_path
        else:
            normalized_path = os.path.join(dataset_base_path, file_path)
        
        # Проверяем существование файла
        if os.path.exists(normalized_path):
            file_list.append(normalized_path)
            file_path_mapping[file_path] = normalized_path
        else:
            print(f"Warning: File not found: {normalized_path}")
    
    if len(file_list) == 0:
        raise ValueError("No valid files found in protocol!")
    
    # Создаем даталоадер с полными путями
    # Используем num_workers=0, чтобы избежать проблем с обработкой ошибок в многопоточности
    dataset = CustomDatasetLoader(file_list, eval_frames=eval_frames, num_eval=num_eval)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    # Извлекаем эмбеддинги
    model.eval()
    embeddings = {}
    
    with torch.no_grad():
        for batch_idx, (audio_batch, file_paths_batch) in enumerate(dataloader):
            # Преобразуем file_paths_batch в список, если это не список
            if isinstance(file_paths_batch, tuple):
                file_paths_batch = list(file_paths_batch)
            elif not isinstance(file_paths_batch, list):
                file_paths_batch = [file_paths_batch]
            
            audio_batch = audio_batch.to(device)
            
            # Обрабатываем стерео аудио (преобразуем в моно)
            # Если последняя размерность равна 2 (стерео), усредняем каналы
            if len(audio_batch.shape) == 4 and audio_batch.shape[-1] == 2:
                # Форма: (batch_size, num_eval, audio_length, 2)
                # Усредняем по последней оси для получения моно
                audio_batch = audio_batch.mean(dim=-1)  # (batch_size, num_eval, audio_length)
            elif len(audio_batch.shape) == 3 and audio_batch.shape[-1] == 2:
                # Форма: (num_eval, audio_length, 2) - случай без батча
                audio_batch = audio_batch.mean(dim=-1)  # (num_eval, audio_length)
                audio_batch = audio_batch.unsqueeze(0)  # (1, num_eval, audio_length)
            
            # Проверяем и исправляем форму тензора
            # audio_batch должен быть (batch_size, num_eval, audio_length)
            if len(audio_batch.shape) == 2:
                # Если форма (num_eval, audio_length), добавляем размерность батча
                audio_batch = audio_batch.unsqueeze(0)  # (1, num_eval, audio_length)
            
            # Проверяем, что тензор имеет правильную форму
            if len(audio_batch.shape) != 3:
                raise ValueError(f"Unexpected audio_batch shape: {audio_batch.shape}, expected (batch_size, num_eval, audio_length)")
            
            # audio_batch shape: (batch_size, num_eval, audio_length)
            batch_size_actual = audio_batch.shape[0]
            num_eval_actual = audio_batch.shape[1]
            
            # Обрабатываем каждый сегмент
            batch_embeddings = []
            for eval_idx in range(num_eval_actual):
                # Модель ожидает тензор формы (batch_size, audio_length)
                audio_segment = audio_batch[:, eval_idx, :]  # (batch_size, audio_length)
                
                # Убеждаемся, что тензор имеет правильную форму (2D)
                if len(audio_segment.shape) == 1:
                    # Если получили 1D тензор, добавляем размерность батча
                    audio_segment = audio_segment.unsqueeze(0)  # (1, audio_length)
                elif len(audio_segment.shape) != 2:
                    raise ValueError(f"Unexpected audio_segment shape: {audio_segment.shape}, expected (batch_size, audio_length)")
                
                # Проверяем, что тензор действительно 2D
                assert len(audio_segment.shape) == 2, f"audio_segment must be 2D, got shape {audio_segment.shape}"
                
                emb = model(audio_segment)  # (batch_size, embedding_dim)
                batch_embeddings.append(emb)
            
            # Усредняем эмбеддинги по сегментам
            batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (num_eval, batch_size, embedding_dim)
            batch_embeddings = torch.mean(batch_embeddings, dim=0)  # (batch_size, embedding_dim)
            
            # Сохраняем эмбеддинги (используем оригинальные пути из протокола)
            for i, normalized_path in enumerate(file_paths_batch):
                # Проверяем, что эмбеддинг не нулевой (файл был успешно загружен)
                # Если эмбеддинг нулевой, пропускаем его
                if torch.all(batch_embeddings[i] == 0):
                    continue
                
                # Находим оригинальный путь из маппинга
                original_path = None
                for orig, norm in file_path_mapping.items():
                    if norm == normalized_path:
                        original_path = orig
                        break
                
                # Если не нашли в маппинге, используем нормализованный путь
                if original_path is None:
                    original_path = normalized_path
                
                embeddings[original_path] = batch_embeddings[i].cpu()
    
    return embeddings


def loadWAV_mono(filename, max_frames, evalmode=True, num_eval=10):
    """
    Загружает WAV файл и преобразует стерео в моно
    """
    import soundfile as sf
    import numpy as np
    import random
    
    try:
        # Загружаем аудио
        audio, sample_rate = sf.read(filename, dtype='float32')
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        raise
    
    # Преобразуем стерео в моно, если необходимо
    if len(audio.shape) == 2:
        # Стерео: форма (audio_length, 2)
        audio = audio.mean(axis=1)  # Усредняем каналы -> (audio_length,)
    elif len(audio.shape) > 2:
        # Многоканальное аудио - берем первый канал
        audio = audio[:, 0]
    
    # Убеждаемся, что audio одномерный массив
    if len(audio.shape) != 1:
        # Если все еще многомерный, берем первый канал или усредняем
        if audio.shape[-1] <= 2:
            audio = audio.mean(axis=-1)
        else:
            audio = audio.flatten()
    
    # Теперь обрабатываем как моно аудио
    max_audio = max_frames * 160 + 240
    audiosize = len(audio)
    
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        # Правильный синтаксис для numpy.pad: кортеж (before, after) для каждой оси
        audio = np.pad(audio, (0, shortage), mode='wrap')
        audiosize = len(audio)
    
    if evalmode:
        if audiosize > max_audio:
            startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            # Если аудио короче max_audio, используем весь файл
            startframe = np.array([0])
    else:
        if audiosize > max_audio:
            startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])
        else:
            startframe = np.array([0])
    
    feats = []
    
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            start_idx = int(asf)
            end_idx = start_idx + max_audio
            if end_idx > audiosize:
                # Если выходим за границы, используем весь доступный аудио
                segment = audio[start_idx:]
                # Дополняем до нужной длины
                if len(segment) < max_audio:
                    segment = np.pad(segment, (0, max_audio - len(segment)), mode='wrap')
            else:
                segment = audio[start_idx:end_idx]
            feats.append(segment)
    
    feat = np.stack(feats, axis=0).astype(np.float32)
    
    return feat


class CustomDatasetLoader:
    """
    Кастомный даталоадер для работы с полными путями к файлам
    """
    def __init__(self, file_list, eval_frames=200, num_eval=5):
        self.file_list = file_list
        self.eval_frames = eval_frames
        self.num_eval = num_eval
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            # Используем функцию, которая обрабатывает стерео
            audio = loadWAV_mono(file_path, self.eval_frames, evalmode=True, num_eval=self.num_eval)
            
            # audio должна быть формы (num_eval, audio_length)
            # Преобразуем в torch tensor
            audio_tensor = torch.FloatTensor(audio)
            return audio_tensor, file_path
        except Exception as e:
            # Если не удалось загрузить файл, возвращаем нулевой тензор
            # Это позволит пропустить проблемные файлы
            print(f"Warning: Could not load {file_path}: {e}")
            # Создаем нулевой тензор правильной формы
            max_audio = self.eval_frames * 160 + 240
            zero_audio = np.zeros((self.num_eval, max_audio), dtype=np.float32)
            audio_tensor = torch.FloatTensor(zero_audio)
            return audio_tensor, file_path


def compute_protocol_scores(embeddings, protocol):
    """
    Вычисляет скоры для протокола
    
    Args:
        embeddings: словарь эмбеддингов {file_path: tensor}
        protocol: список (label, enroll_path, test_path, enroll_year, test_year)
    
    Returns:
        scores: список скоров
        labels: список меток (1/0)
        trials_info: список информации о попытках (для анализа по возрасту)
    """
    scores = []
    labels = []
    trials_info = []
    
    # Нормализуем пути в embeddings для сравнения
    normalized_embeddings = {}
    for path, emb in embeddings.items():
        normalized_path = os.path.normpath(os.path.abspath(path))
        normalized_embeddings[normalized_path] = emb
    
    for label, enroll_path, test_path, enroll_year, test_year in protocol:
        # Нормализуем пути из протокола
        enroll_norm = os.path.normpath(os.path.abspath(enroll_path))
        test_norm = os.path.normpath(os.path.abspath(test_path))
        
        # Проверяем наличие эмбеддингов
        if enroll_norm not in normalized_embeddings or test_norm not in normalized_embeddings:
            # Пробуем найти по базовому имени файла
            enroll_found = False
            test_found = False
            
            for emb_path in normalized_embeddings.keys():
                if enroll_path in emb_path or os.path.basename(enroll_path) in emb_path:
                    enroll_norm = emb_path
                    enroll_found = True
                    break
            
            for emb_path in normalized_embeddings.keys():
                if test_path in emb_path or os.path.basename(test_path) in emb_path:
                    test_norm = emb_path
                    test_found = True
                    break
            
            if not enroll_found or not test_found:
                continue
        
        enroll_emb = normalized_embeddings[enroll_norm].numpy().reshape(1, -1)
        test_emb = normalized_embeddings[test_norm].numpy().reshape(1, -1)
        
        # Вычисляем косинусное сходство
        from sklearn.metrics.pairwise import cosine_similarity
        score = cosine_similarity(enroll_emb, test_emb)[0][0]
        
        scores.append(score)
        labels.append(label)
        trials_info.append({
            'enroll_year': enroll_year,
            'test_year': test_year,
            'year_diff': abs(test_year - enroll_year)
        })
    
    return np.array(scores), np.array(labels), trials_info


def evaluate_protocol(
    model,
    protocol_path: str,
    dataset_base_path: str,
    eval_frames: int = 200,
    device: str = 'cuda'
):
    """
    Полная оценка протокола
    
    Returns:
        eer: равновероятная ошибка
        scores: скоры
        labels: метки
        trials_info: информация о попытках
    """
    # Загружаем протокол
    protocol = load_protocol(protocol_path)
    
    print(f"Loaded protocol with {len(protocol)} trials")
    
    # Извлекаем эмбеддинги
    print("Extracting embeddings...")
    embeddings = extract_embeddings_for_protocol(
        model, protocol, dataset_base_path, eval_frames=eval_frames, device=device
    )
    
    print(f"Extracted embeddings for {len(embeddings)} files")
    
    # Вычисляем скоры
    print("Computing scores...")
    scores, labels, trials_info = compute_protocol_scores(embeddings, protocol)
    
    # Вычисляем EER
    target_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    
    eer, thresh = get_eer(target_scores, impostor_scores)
    
    print(f"EER: {eer:.2f}%")
    print(f"Threshold: {thresh:.4f}")
    print(f"Target trials: {len(target_scores)}")
    print(f"Impostor trials: {len(impostor_scores)}")
    
    return eer, scores, labels, trials_info


def plot_age_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    trials_info: list,
    metadata: dict,
    protocol_name: str,
    output_path: str
):
    """
    Строит графики зависимости работы модели от возраста диктора
    """
    # Группируем попытки по возрасту диктора в момент записи эталона
    age_groups = defaultdict(lambda: {'target_scores': [], 'impostor_scores': []})
    
    for score, label, trial_info in zip(scores, labels, trials_info):
        enroll_year = trial_info['enroll_year']
        
        # Находим возраст диктора (нужно найти speaker_id из пути)
        # Упрощенный подход: используем год записи как прокси для возраста
        # Или группируем по разнице лет
        
        if label == 1:  # таргет
            age_groups[trial_info['year_diff']]['target_scores'].append(score)
        else:  # импостор
            age_groups[trial_info['year_diff']]['impostor_scores'].append(score)
    
    # Строим графики
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # График 1: Распределение скоров по группам разницы лет (таргет)
    ax1 = axes[0, 0]
    year_diffs = sorted([k for k in age_groups.keys() if len(age_groups[k]['target_scores']) > 0])
    target_means = [np.mean(age_groups[yd]['target_scores']) for yd in year_diffs]
    target_stds = [np.std(age_groups[yd]['target_scores']) for yd in year_diffs]
    
    ax1.errorbar(year_diffs, target_means, yerr=target_stds, fmt='o-', label='Target trials')
    ax1.set_xlabel('Year difference (years)')
    ax1.set_ylabel('Mean cosine similarity')
    ax1.set_title(f'Target scores by year difference - {protocol_name}')
    ax1.grid(True)
    ax1.legend()
    
    # График 2: Распределение скоров по группам разницы лет (импостор)
    ax2 = axes[0, 1]
    impostor_means = [np.mean(age_groups[yd]['impostor_scores']) for yd in year_diffs if len(age_groups[yd]['impostor_scores']) > 0]
    impostor_stds = [np.std(age_groups[yd]['impostor_scores']) for yd in year_diffs if len(age_groups[yd]['impostor_scores']) > 0]
    year_diffs_imp = [yd for yd in year_diffs if len(age_groups[yd]['impostor_scores']) > 0]
    
    if len(impostor_means) > 0:
        ax2.errorbar(year_diffs_imp, impostor_means, yerr=impostor_stds, fmt='s-', color='red', label='Impostor trials')
    ax2.set_xlabel('Year difference (years)')
    ax2.set_ylabel('Mean cosine similarity')
    ax2.set_title(f'Impostor scores by year difference - {protocol_name}')
    ax2.grid(True)
    ax2.legend()
    
    # График 3: Гистограмма скоров (таргет vs импостор)
    ax3 = axes[1, 0]
    target_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    
    ax3.hist(target_scores, bins=50, alpha=0.5, label=f'Target (n={len(target_scores)})', density=True)
    ax3.hist(impostor_scores, bins=50, alpha=0.5, label=f'Impostor (n={len(impostor_scores)})', density=True)
    ax3.set_xlabel('Cosine similarity score')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Score distributions - {protocol_name}')
    ax3.legend()
    ax3.grid(True)
    
    # График 4: EER по группам разницы лет
    ax4 = axes[1, 1]
    eers_by_diff = []
    year_diffs_for_eer = []
    
    for year_diff in sorted(set([t['year_diff'] for t in trials_info])):
        mask = np.array([t['year_diff'] == year_diff for t in trials_info])
        if np.sum(mask) == 0:
            continue
        
        group_scores = scores[mask]
        group_labels = labels[mask]
        
        if np.sum(group_labels == 1) > 0 and np.sum(group_labels == 0) > 0:
            group_target = group_scores[group_labels == 1]
            group_impostor = group_scores[group_labels == 0]
            group_eer, _ = get_eer(group_target, group_impostor)
            eers_by_diff.append(group_eer)
            year_diffs_for_eer.append(year_diff)
    
    if len(eers_by_diff) > 0:
        ax4.plot(year_diffs_for_eer, eers_by_diff, 'o-', color='green')
        ax4.set_xlabel('Year difference (years)')
        ax4.set_ylabel('EER (%)')
        ax4.set_title(f'EER by year difference - {protocol_name}')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

