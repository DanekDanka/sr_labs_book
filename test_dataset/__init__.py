"""
Модуль для тестирования системы верификации диктора на датасете с записями разных годов
"""

from .protocol_generator import (
    load_metadata,
    scan_dataset,
    generate_protocol,
    save_protocol,
    load_protocol
)

from .evaluation import (
    evaluate_protocol,
    plot_age_analysis,
    extract_embeddings_for_protocol,
    compute_protocol_scores
)

__all__ = [
    'load_metadata',
    'scan_dataset',
    'generate_protocol',
    'save_protocol',
    'load_protocol',
    'evaluate_protocol',
    'plot_age_analysis',
    'extract_embeddings_for_protocol',
    'compute_protocol_scores',
]

