"""
Генерация протоколов верификации для датасета с записями разных годов
"""

import os
import csv
import random
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np


def parse_date_of_birth(date_str: str) -> int:
    """
    Парсит дату рождения из строки и возвращает год рождения
    """
    if not date_str or date_str.strip() == '':
        return None
    
    # Пробуем разные форматы
    date_str = date_str.strip()
    
    # Формат: "25.04.1946" или "15.4.1949"
    if '.' in date_str:
        parts = date_str.split('.')
        if len(parts) >= 3:
            try:
                year = int(parts[2])
                if 1900 <= year <= 2025:
                    return year
            except:
                pass
    
    # Формат: "5 ноября 1981 г."
    if 'г.' in date_str or 'г' in date_str:
        # Ищем 4-значное число
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', date_str)
        if years:
            try:
                year = int(years[0])
                if 1900 <= year <= 2025:
                    return year
            except:
                pass
    
    return None


def load_metadata(csv_path: str) -> Dict[str, Dict]:
    """
    Загружает метаданные из CSV файла
    Возвращает словарь: {speaker_id: {name, gender, birth_year, ...}}
    """
    metadata = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_id = row['id'].strip()
            birth_year = parse_date_of_birth(row.get('date of birth', ''))
            
            metadata[speaker_id] = {
                'name': row.get('original name', ''),
                'gender': row.get('gender (male or female)', ''),
                'birth_year': birth_year,
                'language': row.get('language', ''),
            }
    
    return metadata


def scan_dataset(dataset_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Сканирует структуру датасета
    Возвращает словарь: {speaker_id: [(year, file_path), ...]}
    """
    dataset_structure = defaultdict(list)
    
    # Проходим по всем папкам с id
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        
        # Пропускаем файлы (например, CSV)
        if not os.path.isdir(item_path):
            continue
        
        # Проверяем, что это папка с id (начинается с id)
        if not item.startswith('id'):
            continue
        
        speaker_id = item
        
        # Проходим по подпапкам с годами
        for year_dir in os.listdir(item_path):
            year_path = os.path.join(item_path, year_dir)
            
            if not os.path.isdir(year_path):
                continue
            
            # Пробуем распарсить год
            try:
                year = int(year_dir)
                if 1900 <= year <= 2025:
                    # Ищем wav файлы в этой папке
                    for wav_file in os.listdir(year_path):
                        if wav_file.endswith('.wav'):
                            wav_path = os.path.join(year_path, wav_file)
                            dataset_structure[speaker_id].append((year, wav_path))
            except ValueError:
                continue
    
    # Сортируем записи по году для каждого диктора
    for speaker_id in dataset_structure:
        dataset_structure[speaker_id].sort(key=lambda x: x[0])
    
    return dict(dataset_structure)


def get_speaker_age_at_year(birth_year: int, recording_year: int) -> int:
    """
    Вычисляет возраст диктора в год записи
    """
    if birth_year is None:
        return None
    return recording_year - birth_year


def generate_protocol(
    dataset_structure: Dict[str, List[Tuple[int, str]]],
    metadata: Dict[str, Dict],
    min_year_diff: int,
    max_year_diff: int,
    target_duration_seconds: int = 10,
    num_target_trials: int = None,
    num_impostor_trials: int = None
) -> List[Tuple[int, str, str, int, int]]:
    """
    Генерирует протокол верификации
    
    Args:
        dataset_structure: структура датасета {speaker_id: [(year, path), ...]}
        metadata: метаданные дикторов
        min_year_diff: минимальная разница в годах между эталоном и тестом
        max_year_diff: максимальная разница в годах между эталоном и тестом
        target_duration_seconds: требуемая длительность речи (секунды)
        num_target_trials: количество таргет-попыток (None = все возможные)
        num_impostor_trials: количество импостор-попыток (None = все возможные)
    
    Returns:
        Список троек: (label, enroll_path, test_path, enroll_year, test_year)
        label: 1 для таргет, 0 для импостор
    """
    protocol = []
    
    # Генерируем таргет-попытки (одинаковый диктор, разные годы)
    target_trials = []
    
    for speaker_id, recordings in dataset_structure.items():
        if len(recordings) < 2:
            continue
        
        # Находим пары записей с нужной разницей в годах
        for i, (year1, path1) in enumerate(recordings):
            for j, (year2, path2) in enumerate(recordings):
                if i >= j:
                    continue
                
                year_diff = abs(year2 - year1)
                
                if min_year_diff <= year_diff <= max_year_diff:
                    # Выбираем более раннюю запись как эталон
                    if year1 < year2:
                        enroll_year, enroll_path = year1, path1
                        test_year, test_path = year2, path2
                    else:
                        enroll_year, enroll_path = year2, path2
                        test_year, test_path = year1, path1
                    
                    target_trials.append((enroll_path, test_path, enroll_year, test_year))
    
    # Ограничиваем количество таргет-попыток
    if num_target_trials is not None and len(target_trials) > num_target_trials:
        target_trials = random.sample(target_trials, num_target_trials)
    
    # Добавляем таргет-попытки в протокол
    for enroll_path, test_path, enroll_year, test_year in target_trials:
        protocol.append((1, enroll_path, test_path, enroll_year, test_year))
    
    # Генерируем импостор-попытки (разные дикторы)
    speaker_ids = list(dataset_structure.keys())
    impostor_trials = []
    
    # Для каждой таргет-попытки создаем несколько импостор-попыток
    num_impostor_per_target = max(1, (num_impostor_trials or len(target_trials)) // max(len(target_trials), 1))
    
    for enroll_path, test_path, enroll_year, test_year in target_trials:
        # Извлекаем speaker_id из пути
        enroll_speaker = None
        for sid in speaker_ids:
            if sid in enroll_path:
                enroll_speaker = sid
                break
        
        if enroll_speaker is None:
            continue
        
        # Находим другого диктора с записями в похожие годы
        for _ in range(num_impostor_per_target):
            other_speaker = random.choice(speaker_ids)
            if other_speaker == enroll_speaker:
                continue
            
            other_recordings = dataset_structure[other_speaker]
            if len(other_recordings) == 0:
                continue
            
            # Выбираем случайную запись другого диктора
            other_year, other_path = random.choice(other_recordings)
            
            impostor_trials.append((enroll_path, other_path, enroll_year, other_year))
    
    # Ограничиваем количество импостор-попыток
    if num_impostor_trials is not None and len(impostor_trials) > num_impostor_trials:
        impostor_trials = random.sample(impostor_trials, num_impostor_trials)
    
    # Добавляем импостор-попытки в протокол
    for enroll_path, test_path, enroll_year, test_year in impostor_trials:
        protocol.append((0, enroll_path, test_path, enroll_year, test_year))
    
    # Перемешиваем протокол
    random.shuffle(protocol)
    
    return protocol


def save_protocol(protocol: List[Tuple[int, str, str, int, int]], output_path: str):
    """
    Сохраняет протокол в файл
    Формат: label enroll_path test_path enroll_year test_year
    Пути сохраняются как абсолютные пути
    """
    with open(output_path, 'w') as f:
        for label, enroll_path, test_path, enroll_year, test_year in protocol:
            # Сохраняем абсолютные пути
            enroll_abs = os.path.abspath(enroll_path) if not os.path.isabs(enroll_path) else enroll_path
            test_abs = os.path.abspath(test_path) if not os.path.isabs(test_path) else test_path
            f.write(f"{label} {enroll_abs} {test_abs} {enroll_year} {test_year}\n")


def load_protocol(protocol_path: str) -> List[Tuple[int, str, str, int, int]]:
    """
    Загружает протокол из файла
    """
    protocol = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                label = int(parts[0])
                enroll_path = parts[1]
                test_path = parts[2]
                enroll_year = int(parts[3])
                test_year = int(parts[4])
                protocol.append((label, enroll_path, test_path, enroll_year, test_year))
    return protocol

