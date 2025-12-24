"""
Оценка качества системы верификации диктора
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt
import soundfile as sf

# Добавляем путь к общим модулям
sys.path.append(os.path.realpath('..'))

from common import get_eer
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
                
                # Модель автоматически применяет предобработку (PreEmphasis + MelSpectrogram) через torchfb
                # в методе forward. Формат: (batch_size, audio_length) -> (batch_size, embedding_dim)
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
    Загружает WAV файл и преобразует стерео в моно.
    Аналогично loadWAV из common/DatasetLoader.py, но с поддержкой стерео.
    Предобработка (PreEmphasis, MelSpectrogram) применяется в модели автоматически.
    """
    import soundfile as sf
    import numpy as np
    import random
    
    max_audio = max_frames * 160 + 240
    
    try:
        # Загружаем аудио (аналогично common/loadWAV)
        audio, sample_rate = sf.read(filename, dtype='float32')
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        raise
    
    # Преобразуем стерео в моно, если необходимо (дополнительная обработка для стерео файлов)
    if len(audio.shape) == 2:
        # Стерео: форма (audio_length, 2)
        audio = audio.mean(axis=1)  # Усредняем каналы -> (audio_length,)
    elif len(audio.shape) > 2:
        # Многоканальное аудио - берем первый канал
        audio = audio[:, 0]
    
    # Убеждаемся, что audio одномерный массив
    if len(audio.shape) != 1:
        if audio.shape[-1] <= 2:
            audio = audio.mean(axis=-1)
        else:
            audio = audio.flatten()
    
    # Теперь обрабатываем как моно аудио (аналогично common/loadWAV)
    audiosize = len(audio)
    
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), mode='wrap')
        audiosize = len(audio)
    
    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])
    
    feats = []
    
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
    
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


def extract_speaker_id_from_path(file_path):
    """
    Извлекает speaker_id из пути к файлу
    Например: /path/to/id00120/2005/00001.wav -> id00120
    """
    # Нормализуем путь
    normalized_path = os.path.normpath(file_path)
    parts = normalized_path.split(os.sep)
    
    # Ищем часть пути, которая начинается с 'id' и содержит цифры
    for part in parts:
        if part.startswith('id'):
            # Проверяем, что после 'id' идут цифры
            if len(part) > 2 and part[2:].isdigit():
                return part
    
    # Альтернативный способ: ищем паттерн idXXXXX в пути
    import re
    match = re.search(r'id\d+', normalized_path)
    if match:
        return match.group(0)
    
    return None


def compute_protocol_scores(embeddings, protocol, metadata=None):
    """
    Вычисляет скоры для протокола
    
    Args:
        embeddings: словарь эмбеддингов {file_path: tensor}
        protocol: список (label, enroll_path, test_path, enroll_year, test_year)
        metadata: словарь метаданных дикторов {speaker_id: {birth_year, ...}}
    
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
        
        # Извлекаем speaker_id и год рождения
        speaker_id = None
        birth_year = None
        
        if metadata:
            # Пробуем извлечь speaker_id из enroll_path (для таргет-попыток это правильный диктор)
            speaker_id = extract_speaker_id_from_path(enroll_path)
            
            # Если не получилось из enroll_path, пробуем test_path (для таргет-попыток это тот же диктор)
            if speaker_id is None:
                speaker_id = extract_speaker_id_from_path(test_path)
            
            # Если нашли speaker_id, получаем год рождения из метаданных
            if speaker_id and speaker_id in metadata:
                birth_year = metadata[speaker_id].get('birth_year')
            elif speaker_id:
                # Пробуем найти похожий speaker_id (на случай опечаток или различий в формате)
                for sid in metadata.keys():
                    if sid.lower() == speaker_id.lower() or sid.replace('_', '') == speaker_id.replace('_', ''):
                        speaker_id = sid
                        birth_year = metadata[sid].get('birth_year')
                        break
        
        scores.append(score)
        labels.append(label)
        trials_info.append({
            'enroll_year': enroll_year,
            'test_year': test_year,
            'year_diff': abs(test_year - enroll_year),
            'speaker_id': speaker_id,
            'birth_year': birth_year,
            'enroll_path': enroll_path,  # Сохраняем пути для отладки
            'test_path': test_path
        })
    
    return np.array(scores), np.array(labels), trials_info


def evaluate_protocol(
    model,
    protocol_path: str,
    dataset_base_path: str,
    eval_frames: int = 200,
    device: str = 'cuda',
    metadata: dict = None
):
    """
    Полная оценка протокола
    
    Args:
        model: обученная модель
        protocol_path: путь к файлу протокола
        dataset_base_path: базовый путь к датасету
        eval_frames: количество фреймов для оценки
        device: устройство (cuda/cpu)
        metadata: словарь метаданных дикторов (опционально)
    
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
    # Если метаданные не переданы, пробуем загрузить их
    if metadata is None:
        try:
            from protocol_generator import load_metadata
            # Пробуем найти CSV файл в директории датасета
            csv_path = os.path.join(dataset_base_path, 'Каталог личностей для доп. задания №2 (2025) - Каталог личностей.csv')
            if os.path.exists(csv_path):
                metadata = load_metadata(csv_path)
        except:
            pass
    
    scores, labels, trials_info = compute_protocol_scores(embeddings, protocol, metadata)
    
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
        
        group_target = group_scores[group_labels == 1]
        group_impostor = group_scores[group_labels == 0]
        
        # Проверяем, что есть достаточно точек для вычисления EER
        # Нужно минимум 2 точки для каждой группы
        if len(group_target) >= 2 and len(group_impostor) >= 2:
            try:
                group_eer, _ = get_eer(group_target, group_impostor)
                if not np.isnan(group_eer) and np.isfinite(group_eer):
                    eers_by_diff.append(group_eer)
                    year_diffs_for_eer.append(year_diff)
            except (ValueError, Exception) as e:
                # Пропускаем группы, для которых не удалось вычислить EER
                continue
    
    if len(eers_by_diff) > 0:
        ax4.plot(year_diffs_for_eer, eers_by_diff, 'o-', color='green')
        ax4.set_xlabel('Year difference (years)')
        ax4.set_ylabel('EER (%)')
        ax4.set_title(f'EER by year difference - {protocol_name}')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'Not enough data for EER calculation', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'EER by year difference - {protocol_name}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_protocol3_heatmap(results_file: str, output_path: str):
    """
    Строит тепловую карту для протокола 3:
    - Ось X: разница лет между записями (year_diff)
    - Ось Y: возраст диктора в момент тестовой записи (test_age)
    - Цвет: probability (вероятность, что это один и тот же диктор)
    
    Args:
        results_file: путь к CSV файлу с результатами протокола 3
        output_path: путь для сохранения графика
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas is not installed. Please install it: pip install pandas")
        return
    
    # Читаем данные
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return
    
    if len(df) == 0:
        print(f"Warning: No data in {results_file}")
        return
    
    # Преобразуем типы данных
    df['test_age'] = pd.to_numeric(df['test_age'], errors='coerce')
    df['year_diff'] = pd.to_numeric(df['year_diff'], errors='coerce')
    df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
    
    # Удаляем строки с пустыми значениями
    df_plot = df[df['test_age'].notna() & df['year_diff'].notna() & df['probability'].notna()].copy()
    
    if len(df_plot) == 0:
        print(f"Warning: No valid data points in {results_file}")
        return
    
    print(f"Data points for heatmap: {len(df_plot)}")
    
    # Строим тепловую карту
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Интерполируем для плавного градиента
    from scipy.interpolate import griddata
    
    years_diff = df_plot['year_diff'].values
    ages = df_plot['test_age'].values
    probabilities = df_plot['probability'].values
    
    if len(probabilities) >= 3:
        points = np.array([[yd, age] for yd, age in zip(years_diff, ages)])
        values = probabilities
        
        years_diff_unique = sorted(set(years_diff))
        ages_unique = sorted(set(ages))
        
        xi = np.linspace(min(years_diff_unique), max(years_diff_unique), 100)
        yi = np.linspace(min(ages_unique), max(ages_unique), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        try:
            Zi = griddata(points, values, (Xi, Yi), method='linear', fill_value=np.nan)
            if not np.all(np.isnan(Zi)):
                im = ax.contourf(Xi, Yi, Zi, levels=50, cmap='viridis', alpha=0.7, extend='both')
                contours = ax.contour(Xi, Yi, Zi, levels=20, colors='black', alpha=0.3, linewidths=0.5)
                ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        except Exception as e:
            print(f"Warning: Interpolation failed: {e}")
            Zi = None
    else:
        Zi = None
    
    # Точки данных
    scatter = ax.scatter(years_diff, ages, c=probabilities, cmap='viridis',
                        s=150, edgecolors='black', linewidth=1.5, alpha=0.9, zorder=5,
                        vmin=min(probabilities), vmax=max(probabilities))
    
    # Цветовая шкала
    if Zi is not None and not np.all(np.isnan(Zi)):
        cbar = plt.colorbar(im, ax=ax, label='Probability (Same Speaker)', pad=0.02)
    else:
        cbar = plt.colorbar(scatter, ax=ax, label='Probability (Same Speaker)', pad=0.02)
    cbar.set_label('Probability (Same Speaker)', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Year Difference (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speaker Age at Test Recording', fontsize=14, fontweight='bold')
    ax.set_title('Protocol 3: Probability of Same Speaker\n(Oldest Recording vs Newer Recordings)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    plt.close()


def evaluate_speaker_protocol(
    model,
    dataset_path: str,
    metadata: dict,
    min_year_diff: int,
    max_year_diff: int,
    eval_frames: int = 200,
    device: str = 'cuda',
    output_file: str = None
):
    """
    Для каждого диктора:
    1. Находит самую старую запись как эталон
    2. Ищет более новые записи с разницей в годах между min_year_diff и max_year_diff
    3. Вычисляет EER для каждого диктора отдельно
    4. Сохраняет результаты в файл
    
    Args:
        model: обученная модель
        dataset_path: путь к датасету
        metadata: словарь метаданных дикторов
        min_year_diff: минимальная разница в годах
        max_year_diff: максимальная разница в годах
        eval_frames: количество фреймов для оценки
        device: устройство (cuda/cpu)
        output_file: путь к файлу для сохранения результатов (CSV)
    
    Returns:
        results: список словарей с результатами для каждого диктора
    """
    import csv
    
    if metadata is None or len(metadata) == 0:
        print("Warning: No metadata provided. Skipping.")
        return []
    
    # Сканируем датасет
    from protocol_generator import scan_dataset
    dataset_structure = scan_dataset(dataset_path)
    
    print(f"Found {len(dataset_structure)} speakers in dataset")
    
    # Для каждого диктора находим самую старую запись и подходящие более новые
    speaker_data = []
    
    for speaker_id, recordings in dataset_structure.items():
        if speaker_id not in metadata:
            continue
        
        birth_year = metadata[speaker_id].get('birth_year')
        if birth_year is None:
            continue
        
        # Сортируем записи по году
        sorted_recordings = sorted(recordings, key=lambda x: x[0])  # x[0] - это год
        
        if len(sorted_recordings) < 2:
            continue  # Нужно минимум 2 записи
        
        # Самая старая запись - эталон
        oldest_year, oldest_path = sorted_recordings[0]
        oldest_age = oldest_year - birth_year
        
        # Ищем более новые записи с нужной разницей в годах
        matching_recordings = []
        for newer_year, newer_path in sorted_recordings[1:]:
            year_diff = newer_year - oldest_year
            if min_year_diff <= year_diff <= max_year_diff:
                newer_age = newer_year - birth_year
                matching_recordings.append((newer_year, newer_path, newer_age, year_diff))
        
        if len(matching_recordings) == 0:
            continue  # Нет подходящих записей
        
        speaker_data.append({
            'speaker_id': speaker_id,
            'birth_year': birth_year,
            'oldest_year': oldest_year,
            'oldest_path': oldest_path,
            'oldest_age': oldest_age,
            'matching_recordings': matching_recordings
        })
    
    print(f"Speakers with matching recordings: {len(speaker_data)}")
    
    if len(speaker_data) == 0:
        print("Warning: No speakers with matching recordings.")
        return []
    
    # Извлекаем эмбеддинги для всех записей
    print("Extracting embeddings for all recordings...")
    all_files = []
    
    for sd in speaker_data:
        all_files.append(sd['oldest_path'])
        for newer_year, newer_path, newer_age, year_diff in sd['matching_recordings']:
            all_files.append(newer_path)
    
    # Удаляем дубликаты
    all_files = list(set(all_files))
    print(f"Total unique files to process: {len(all_files)}")
    
    # Извлекаем эмбеддинги
    embeddings = {}
    dataset_loader = CustomDatasetLoader(all_files, eval_frames=eval_frames, num_eval=5)
    dataloader = DataLoader(dataset_loader, batch_size=1, shuffle=False, num_workers=0)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (audio_batch, file_paths) in enumerate(dataloader):
            if audio_batch is None:
                continue
            
            if audio_batch.sum() == 0:
                continue
            
            try:
                batch_embeddings = []
                for eval_idx in range(audio_batch.shape[1]):
                    audio_segment = audio_batch[:, eval_idx, :]
                    
                    if audio_segment.dim() > 2:
                        audio_segment = audio_segment.squeeze()
                    
                    if audio_segment.dim() != 2:
                        continue
                    
                    audio_segment = audio_segment.to(device)
                    emb = model(audio_segment)
                    emb = F.normalize(emb, p=2, dim=1)
                    batch_embeddings.append(emb.cpu())
                
                if len(batch_embeddings) == 0:
                    continue
                
                avg_embedding = torch.mean(torch.stack(batch_embeddings), dim=0)
                
                for file_path in file_paths:
                    embeddings[file_path] = avg_embedding
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    print(f"Extracted embeddings for {len(embeddings)} files")
    
    # Для каждого диктора вычисляем EER
    results = []
    
    # Устанавливаем seed один раз для всех дикторов
    import random
    random.seed(42)
    
    for sd in speaker_data:
        speaker_id = sd['speaker_id']
        birth_year = sd['birth_year']
        oldest_path = sd['oldest_path']
        oldest_year = sd['oldest_year']
        oldest_age = sd['oldest_age']
        
        if oldest_path not in embeddings:
            continue
        
        oldest_emb = embeddings[oldest_path]
        
        # Собираем таргет-скоры (пары того же диктора)
        target_scores = []
        
        for newer_year, newer_path, newer_age, year_diff in sd['matching_recordings']:
            if newer_path not in embeddings:
                continue
            
            newer_emb = embeddings[newer_path]
            
            # Вычисляем косинусное сходство
            from sklearn.metrics.pairwise import cosine_similarity
            score = cosine_similarity(
                oldest_emb.numpy().reshape(1, -1),
                newer_emb.numpy().reshape(1, -1)
            )[0][0]
            
            target_scores.append(score)
            
            # Сохраняем результат для этого диктора и этой пары
            results.append({
                'speaker_id': speaker_id,
                'enroll_age': oldest_age,
                'test_age': newer_age,
                'year_diff': year_diff,
                'score': score
            })
        
        # Если нет таргет-скоров, пропускаем
        if len(target_scores) == 0:
            continue
        
        # Создаем импостор-скоры (пары разных дикторов)
        impostor_scores = []
        
        for other_sd in speaker_data:
            if other_sd['speaker_id'] == speaker_id:
                continue
            
            other_oldest_path = other_sd['oldest_path']
            if other_oldest_path not in embeddings:
                continue
            
            if len(other_sd['matching_recordings']) == 0:
                continue
            
            # Берем все подходящие записи другого диктора (или несколько случайных, если их слишком много)
            other_matching = other_sd['matching_recordings']
            if len(other_matching) > 5:
                # Если записей много, берем случайные 5
                other_matching = random.sample(other_matching, 5)
            
            for newer_year, newer_path, newer_age, year_diff in other_matching:
                if newer_path not in embeddings:
                    continue
                
                newer_emb = embeddings[newer_path]
                
                from sklearn.metrics.pairwise import cosine_similarity
                score = cosine_similarity(
                    oldest_emb.numpy().reshape(1, -1),
                    newer_emb.numpy().reshape(1, -1)
                )[0][0]
                impostor_scores.append(score)
        
        # Вычисляем EER для этого диктора
        speaker_eer = None
        if len(target_scores) >= 2 and len(impostor_scores) >= 2:
            try:
                eer, _ = get_eer(np.array(target_scores), np.array(impostor_scores))
                if not np.isnan(eer) and np.isfinite(eer):
                    speaker_eer = eer
            except:
                pass
        
        # Добавляем EER ко всем результатам этого диктора
        for result in results:
            if result['speaker_id'] == speaker_id:
                result['eer'] = speaker_eer
    
    # Сохраняем результаты в файл
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['speaker_id', 'enroll_age', 'test_age', 'year_diff', 'score', 'eer']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'speaker_id': result.get('speaker_id', ''),
                    'enroll_age': result.get('enroll_age', ''),
                    'test_age': result.get('test_age', ''),
                    'year_diff': result.get('year_diff', ''),
                    'score': f"{result.get('score', 0):.6f}",
                    'eer': f"{result.get('eer', 0):.2f}" if result.get('eer') is not None else ''
                })
        
        print(f"Saved results to {output_file}")
    
    return results


def evaluate_speaker_protocol_all_newer(
    model,
    dataset_path: str,
    metadata: dict,
    eval_frames: int = 200,
    device: str = 'cuda',
    output_file: str = None
):
    """
    Протокол 3: Для каждого диктора берется самая старая запись как эталон
    и сравнивается со всеми более новыми записями того же диктора.
    
    Args:
        model: обученная модель
        dataset_path: путь к датасету
        metadata: словарь метаданных дикторов
        eval_frames: количество фреймов для оценки
        device: устройство (cuda/cpu)
        output_file: путь к файлу для сохранения результатов (CSV)
    
    Returns:
        results: список словарей с результатами для каждого диктора
    """
    import csv
    
    if metadata is None or len(metadata) == 0:
        print("Warning: No metadata provided. Skipping.")
        return []
    
    # Сканируем датасет
    from protocol_generator import scan_dataset
    dataset_structure = scan_dataset(dataset_path)
    
    print(f"Found {len(dataset_structure)} speakers in dataset")
    
    # Для каждого диктора находим самую старую запись и все более новые
    speaker_data = []
    
    for speaker_id, recordings in dataset_structure.items():
        if speaker_id not in metadata:
            continue
        
        birth_year = metadata[speaker_id].get('birth_year')
        if birth_year is None:
            continue
        
        # Сортируем записи по году
        sorted_recordings = sorted(recordings, key=lambda x: x[0])
        
        if len(sorted_recordings) < 2:
            continue
        
        # Самая старая запись - эталон
        oldest_year, oldest_path = sorted_recordings[0]
        oldest_age = oldest_year - birth_year
        
        # Все более новые записи - тесты
        matching_recordings = []
        for newer_year, newer_path in sorted_recordings[1:]:
            newer_age = newer_year - birth_year
            year_diff = newer_year - oldest_year
            matching_recordings.append((newer_year, newer_path, newer_age, year_diff))
        
        speaker_data.append({
            'speaker_id': speaker_id,
            'birth_year': birth_year,
            'oldest_year': oldest_year,
            'oldest_path': oldest_path,
            'oldest_age': oldest_age,
            'matching_recordings': matching_recordings
        })
    
    print(f"Speakers with at least 2 recordings: {len(speaker_data)}")
    
    if len(speaker_data) == 0:
        print("Warning: No speakers with at least 2 recordings.")
        return []
    
    # Извлекаем эмбеддинги для всех записей
    print("Extracting embeddings for all recordings...")
    all_files = []
    
    for sd in speaker_data:
        all_files.append(sd['oldest_path'])
        for newer_year, newer_path, newer_age, year_diff in sd['matching_recordings']:
            all_files.append(newer_path)
    
    all_files = list(set(all_files))
    print(f"Total unique files to process: {len(all_files)}")
    
    # Извлекаем эмбеддинги
    embeddings = {}
    dataset_loader = CustomDatasetLoader(all_files, eval_frames=eval_frames, num_eval=5)
    dataloader = DataLoader(dataset_loader, batch_size=1, shuffle=False, num_workers=0)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (audio_batch, file_paths) in enumerate(dataloader):
            if audio_batch is None:
                continue
            
            if audio_batch.sum() == 0:
                continue
            
            try:
                batch_embeddings = []
                for eval_idx in range(audio_batch.shape[1]):
                    audio_segment = audio_batch[:, eval_idx, :]
                    
                    if audio_segment.dim() > 2:
                        audio_segment = audio_segment.squeeze()
                    
                    if audio_segment.dim() != 2:
                        continue
                    
                    audio_segment = audio_segment.to(device)
                    emb = model(audio_segment)
                    emb = F.normalize(emb, p=2, dim=1)
                    batch_embeddings.append(emb.cpu())
                
                if len(batch_embeddings) == 0:
                    continue
                
                avg_embedding = torch.mean(torch.stack(batch_embeddings), dim=0)
                
                for file_path in file_paths:
                    embeddings[file_path] = avg_embedding
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    print(f"Extracted embeddings for {len(embeddings)} files")
    
    # Для каждого диктора вычисляем вероятность/уверенность, что более новые записи принадлежат тому же диктору
    results = []
    
    for sd in speaker_data:
        speaker_id = sd['speaker_id']
        birth_year = sd['birth_year']
        oldest_path = sd['oldest_path']
        oldest_year = sd['oldest_year']
        oldest_age = sd['oldest_age']
        
        if oldest_path not in embeddings:
            continue
        
        oldest_emb = embeddings[oldest_path]
        
        # Вычисляем вероятность/уверенность, что более новые записи принадлежат тому же диктору
        for newer_year, newer_path, newer_age, year_diff in sd['matching_recordings']:
            if newer_path not in embeddings:
                continue
            
            newer_emb = embeddings[newer_path]
            
            from sklearn.metrics.pairwise import cosine_similarity
            # Cosine similarity как мера уверенности (чем выше, тем больше уверенность)
            # Для нормализованных эмбеддингов cosine similarity в диапазоне [-1, 1]
            # Обычно для нормализованных эмбеддингов это [0, 1], где 1 = полная уверенность
            confidence_score = cosine_similarity(
                oldest_emb.numpy().reshape(1, -1),
                newer_emb.numpy().reshape(1, -1)
            )[0][0]
            
            # Нормализуем в диапазон [0, 1] для интерпретации как вероятности
            # (cosine similarity уже обычно в [0, 1] для нормализованных векторов)
            # Если score отрицательный, устанавливаем в 0
            probability = max(0.0, min(1.0, (confidence_score + 1) / 2)) if confidence_score < 0 else confidence_score
            
            results.append({
                'speaker_id': speaker_id,
                'enroll_age': oldest_age,
                'test_age': newer_age,
                'year_diff': year_diff,
                'score': confidence_score,
                'probability': probability
            })
    
    # Сохраняем результаты
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['speaker_id', 'enroll_age', 'test_age', 'year_diff', 'score', 'probability']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'speaker_id': result.get('speaker_id', ''),
                    'enroll_age': result.get('enroll_age', ''),
                    'test_age': result.get('test_age', ''),
                    'year_diff': result.get('year_diff', ''),
                    'score': f"{result.get('score', 0):.6f}",
                    'probability': f"{result.get('probability', 0):.6f}"
                })
        
        print(f"Saved results to {output_file}")
    
    return results


def plot_protocol_results(results_file: str, output_path: str, protocol_name: str = ""):
    """
    Читает результаты из CSV файла и строит графики
    
    Args:
        results_file: путь к CSV файлу с результатами
        output_path: путь для сохранения графиков
        protocol_name: название протокола для заголовков
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas is not installed. Please install it: pip install pandas")
        return
    
    # Читаем данные
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return
    
    if len(df) == 0:
        print(f"Warning: No data in {results_file}")
        return
    
    # Преобразуем типы данных
    df['enroll_age'] = pd.to_numeric(df['enroll_age'], errors='coerce')
    df['test_age'] = pd.to_numeric(df['test_age'], errors='coerce')
    df['year_diff'] = pd.to_numeric(df['year_diff'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # Проверяем, есть ли колонка 'probability' (для протокола 3) или 'eer' (для протоколов 1 и 2)
    use_probability = 'probability' in df.columns
    use_eer = 'eer' in df.columns
    
    if use_probability:
        df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
        df_plot = df[df['probability'].notna()].copy()
        metric_name = 'Probability'
        metric_col = 'probability'
        ylabel = 'Probability (Same Speaker)'
    elif use_eer:
        df['eer'] = pd.to_numeric(df['eer'], errors='coerce')
        df_plot = df[df['eer'].notna()].copy()
        metric_name = 'EER'
        metric_col = 'eer'
        ylabel = 'EER (%)'
    else:
        print(f"Warning: No 'probability' or 'eer' column in {results_file}")
        return
    
    if len(df_plot) == 0:
        print(f"Warning: No valid {metric_name} values in {results_file}")
        return
    
    # Строим графики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Metric vs Year Difference
    ax1 = axes[0, 0]
    ax1.scatter(df_plot['year_diff'], df_plot[metric_col], alpha=0.6, s=50)
    ax1.set_xlabel('Year Difference', fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax1.set_title(f'{metric_name} vs Year Difference - {protocol_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Metric vs Test Age
    ax2 = axes[0, 1]
    ax2.scatter(df_plot['test_age'], df_plot[metric_col], alpha=0.6, s=50)
    ax2.set_xlabel('Test Age', fontsize=12, fontweight='bold')
    ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax2.set_title(f'{metric_name} vs Test Age - {protocol_name}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # График 3: Metric vs Enroll Age
    ax3 = axes[1, 0]
    ax3.scatter(df_plot['enroll_age'], df_plot[metric_col], alpha=0.6, s=50)
    ax3.set_xlabel('Enroll Age', fontsize=12, fontweight='bold')
    ax3.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax3.set_title(f'{metric_name} vs Enroll Age - {protocol_name}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # График 4: Распределение метрики
    ax4 = axes[1, 1]
    ax4.hist(df_plot[metric_col], bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel(ylabel, fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title(f'{metric_name} Distribution - {protocol_name}', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved plots to {output_path}")
    plt.close()

