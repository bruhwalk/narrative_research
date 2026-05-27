"""
Скрипт для вычисления метрик сравнения разметок и создания графиков.

Сравнивает разметки с золотым стандартом (goldenset):
1. Обычная разметка (без контекста) — LLm markup.xlsx (слот 3)
2. Обогащённая разметка llm_context — goldenset_remarked_llm_context_*.xlsx (слот 4)
3. Обогащённая разметка llm_context_features — goldenset_remarked_llm_context_features_*.xlsx (слот 4)
4. Обогащённая разметка llm_context_features_full — goldenset_remarked_llm_context_features_full_*.xlsx (слот 4)

Золотой стандарт: golden_set_v1.csv

Результаты выводятся в 3 отдельные папки для каждого типа обогащения контекста:
- metrics_plots/llm_context/
- metrics_plots/llm_context_features/
- metrics_plots/llm_context_features_full/
"""

import os
import sys
import glob
import argparse
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ==================== КОНФИГУРАЦИЯ ====================

# Входные файлы
# Золотой стандарт - это golden_set_v1.csv с ЧЕЛОВЕЧЕСКОЙ разметкой
GOLDENSET_CSV = 'golden_set_v1.csv'
WITHOUT_CONTEXT_XLSX = 'LLm markup without context.xlsx'

# Параметры разметок
WITHOUT_CONTEXT_SLOT = 3  # Слот разметки без контекста
WITH_CONTEXT_SLOT = 4     # Слот разметки с контекстом

# Типы контекста для сравнения
CONTEXT_TYPES = [
    'llm_context',
    'llm_context_features',
    'llm_context_features_full'
]

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

_base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


def col(name: str, idx: int) -> str:
    """Форматирование имени столбца с номером слота."""
    return f'{name} ({idx})'


def _strip(x: Any) -> str:
    """Очистка значения от пробелов и преобразование в строку."""
    if x is None:
        return ''
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


# ==================== МЕТРИКИ ====================

def safe_f1_score(y_true, y_pred, average='binary'):
    """Вычисление F1 score с обработкой ошибок."""
    from sklearn.metrics import f1_score as sklearn_f1
    try:
        return sklearn_f1(y_true, y_pred, average=average)
    except ValueError:
        return 0.0
    except Exception:
        return 0.0


def safe_precision_score(y_true, y_pred, average='binary'):
    """Вычисление Precision с обработкой ошибок."""
    from sklearn.metrics import precision_score as sklearn_precision
    try:
        return sklearn_precision(y_true, y_pred, average=average)
    except ValueError:
        return 0.0
    except Exception:
        return 0.0


def safe_recall_score(y_true, y_pred, average='binary'):
    """Вычисление Recall с обработкой ошибок."""
    from sklearn.metrics import recall_score as sklearn_recall
    try:
        return sklearn_recall(y_true, y_pred, average=average)
    except ValueError:
        return 0.0
    except Exception:
        return 0.0


def safe_cohen_kappa(y_true, y_pred):
    """Вычисление Cohen's Kappa с обработкой ошибок."""
    from sklearn.metrics import cohen_kappa_score as sklearn_kappa
    try:
        return sklearn_kappa(y_true, y_pred)
    except Exception:
        return 0.0


def safe_fbeta_score(y_true, y_pred, beta=2.0, average='binary'):
    """
    Вычисление F-beta score с обработкой ошибок.

    F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    При beta=2 recall важнее precision в 2 раза.
    """
    from sklearn.metrics import fbeta_score as sklearn_fbeta
    try:
        return sklearn_fbeta(y_true, y_pred, beta=beta, average=average)
    except ValueError:
        return 0.0
    except Exception:
        return 0.0


def safe_f2_score(y_true, y_pred, average='binary'):
    """Вычисление F2 score с обработкой ошибок (beta=2)."""
    return safe_fbeta_score(y_true, y_pred, beta=2.0, average=average)


def compute_metrics_for_field(
    df: pd.DataFrame,
    gold_field: str,
    markup_field: str,
    transform_gold,
    transform_markup
) -> Tuple[Optional[Dict[str, float]], int]:
    """
    Вычисление метрик для одного поля.

    Args:
        df: Объединённый DataFrame (после merge по message_id)
        gold_field: Имя столбца с золотым стандартом
        markup_field: Имя столбца с разметкой LLM
        transform_gold: Функция трансформации золотого стандарта
        transform_markup: Функция трансформации разметки

    Возвращает словарь метрик и количество сэмплов.
    """
    if markup_field not in df.columns:
        return None, 0

    # Маска строк с данными
    mask = (~df[markup_field].isna()) & (df[markup_field] != '')

    if gold_field in df.columns:
        mask = mask & (~df[gold_field].isna())

    if mask.sum() == 0:
        return None, 0

    try:
        # Применяем трансформации
        y_true = df.loc[mask, gold_field].apply(transform_gold).values
        y_pred = df.loc[mask, markup_field].apply(transform_markup).values

        if len(y_true) == 0:
            return None, 0

        # Для бинарных и мультиклассовых метрик
        unique_true = len(np.unique(y_true))
        unique_pred = len(np.unique(y_pred))

        if unique_true < 2 or unique_pred < 2:
            # Только один класс — метрики не имеют смысла
            return None, 0

        # Проверяем является ли классификация бинарной (0 и 1)
        is_binary = (unique_true == 2 and unique_pred == 2 and
                     set(np.unique(y_true)) == {0, 1} and
                     set(np.unique(y_pred)) == {0, 1})

        if is_binary:
            # Бинарная классификация
            f1 = safe_f1_score(y_true, y_pred, average='binary')
            precision = safe_precision_score(y_true, y_pred, average='binary')
            recall = safe_recall_score(y_true, y_pred, average='binary')
            f2 = safe_f2_score(y_true, y_pred, average='binary')
        else:
            # Мультиклассовая - используем weighted
            f1 = safe_f1_score(y_true, y_pred, average='weighted')
            precision = safe_precision_score(y_true, y_pred, average='weighted')
            recall = safe_recall_score(y_true, y_pred, average='weighted')
            # F2 для мультиклассовой - вычисляем через weighted average
            f2 = safe_f2_score(y_true, y_pred, average='weighted')

        kappa = safe_cohen_kappa(y_true, y_pred)

        return {
            'f1': float(f1),
            'f2': float(f2),
            'precision': float(precision),
            'recall': float(recall),
            'kappa': float(kappa)
        }, int(len(y_true))

    except Exception as e:
        print(f'    Ошибка: {e}')
        return None, 0


def gold_to_narrative(x):
    """Преобразование золотого стандарта economic_narrative в 0/1."""
    try:
        val = float(x)
        # Золотой стандарт содержит 0 или 1
        return 1 if val >= 0.5 else 0
    except (ValueError, TypeError):
        s = str(x).strip()
        return 1 if s == 'Да' else 0


def gold_to_strength(x):
    """
    Преобразование золотого стандарта narrative_strength в 1-3.
    
    Золотой стандарт: [0.0 .. 1.0] -> маппим на 1, 2, 3
    0.0-0.33 -> 1 (слабый)
    0.34-0.66 -> 2 (средний)
    0.67-1.0 -> 3 (сильный)
    """
    try:
        val = float(x)
        if val <= 0.33:
            return 1
        elif val <= 0.66:
            return 2
        else:
            return 3
    except (ValueError, TypeError):
        return 1


def gold_to_effect(x):
    """
    Преобразование золотого стандарта economic_effect в -2..2.
    
    Золотой стандарт: [-1.0 .. 1.0] -> маппим на -2, -1, 0, 1, 2
    -1.0 .. -0.6 -> -2 (сильно негативный)
    -0.6 .. -0.2 -> -1 (негативный)
    -0.2 .. 0.2 -> 0 (нейтральный)
    0.2 .. 0.6 -> 1 (позитивный)
    0.6 .. 1.0 -> 2 (сильно позитивный)
    """
    try:
        val = float(x)
        if val <= -0.6:
            return -2
        elif val <= -0.2:
            return -1
        elif val <= 0.2:
            return 0
        elif val <= 0.6:
            return 1
        else:
            return 2
    except (ValueError, TypeError):
        return 0


def gold_to_resonance(x):
    """
    Преобразование золотого стандарта information_resonance в 1-3.
    
    Золотой стандарт: [0.0 .. 1.0] -> маппим на 1, 2, 3
    0.0-0.33 -> 1 (слабый резонанс)
    0.34-0.66 -> 2 (средний резонанс)
    0.67-1.0 -> 3 (сильный резонанс)
    """
    try:
        val = float(x)
        if val <= 0.33:
            return 1
        elif val <= 0.66:
            return 2
        else:
            return 3
    except (ValueError, TypeError):
        return 1


def markup_to_narrative(x):
    """Преобразование разметки economic_narrative в 0/1."""
    s = str(x).strip()
    return 1 if s == 'Да' else 0


def markup_to_int(x):
    """Преобразование разметки в целое число."""
    try:
        return int(float(str(x).strip()))
    except (ValueError, TypeError):
        return 0


def compare_markups(
    df_gold: pd.DataFrame,
    df_markup: pd.DataFrame,
    markup_slot: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    Сравнение разметки с золотым стандартом.

    Соединяет данные по message_id и вычисляет метрики.
    """
    all_metrics = {}

    # Соединяем по message_id с явными суффиксами
    if 'message_id' in df_gold.columns and 'message_id' in df_markup.columns:
        df_merged = pd.merge(
            df_gold[['message_id', 'economic_narrative', 'narrative_strength', 
                     'information_resonance', 'economic_effect']],
            df_markup,
            on='message_id',
            how='inner',
            suffixes=('_gold', '_llm')
        )
        print(f'    Соединено по message_id: {len(df_merged)} строк из {len(df_gold)} золотых и {len(df_markup)} разметки')
    else:
        # Если нет message_id, используем простой вариант (предполагаем одинаковый порядок)
        print('    WARNING: message_id не найден, используем простой merge по индексу')
        min_len = min(len(df_gold), len(df_markup))
        df_merged = pd.concat([
            df_gold[['economic_narrative', 'narrative_strength', 
                     'information_resonance', 'economic_effect']].iloc[:min_len].reset_index(drop=True),
            df_markup.iloc[:min_len].reset_index(drop=True)
        ], axis=1)

    # Поля для сравнения - используем суффиксы _gold и _llm
    fields = [
        ('Экономический нарратив', 'economic_narrative',
         gold_to_narrative, markup_to_narrative),
        ('Сила нарратива', 'narrative_strength',
         gold_to_strength, markup_to_int),
        ('Экономический эффект', 'economic_effect',
         gold_to_effect, markup_to_int),
        ('Информационный резонанс', 'information_resonance',
         gold_to_resonance, markup_to_int),
    ]

    for field_name, field_key, transform_gold, transform_markup in fields:
        # После merge с suffixes столбцы называются field_gold и field_llm
        gold_col = f'{field_key}_gold'
        markup_col = col(field_name, markup_slot)
        
        # Проверяем наличие столбцов
        if gold_col not in df_merged.columns:
            # Пробуем без суффикса (если merge не сработал)
            gold_col = field_key
            
        metrics, samples = compute_metrics_for_field(
            df_merged,
            gold_col, markup_col,
            transform_gold, transform_markup
        )

        all_metrics[field_key] = {
            'metrics': metrics,
            'samples': samples
        }

    return all_metrics


def compare_all_contexts(
    df_gold: pd.DataFrame,
    df_without: pd.DataFrame,
    context_files: Dict[str, str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Сравнение всех типов контекста с золотым стандартом.

    Args:
        df_gold: Золотой стандарт
        df_without: Разметка без контекста
        context_files: Словарь {context_type: filepath}

    Returns:
        Метрики для каждого типа контекста
    """
    all_results = {}

    # Метрики для разметки без контекста
    print(f'\n=== Разметка без контекста ===')
    metrics_without = compare_markups(
        df_gold, df_without,
        markup_slot=WITHOUT_CONTEXT_SLOT
    )
    all_results['without_context'] = {
        field_key: data['metrics'] if data['metrics'] else {}
        for field_key, data in metrics_without.items()
    }
    all_results['without_context_samples'] = {
        field_key: data['samples']
        for field_key, data in metrics_without.items()
    }

    # Метрики для каждого типа контекста
    for context_type, filepath in context_files.items():
        print(f'\n=== Контекст: {context_type} ===')
        print(f'Файл: {filepath}')

        if filepath.endswith('.csv'):
            df_with = pd.read_csv(filepath, encoding='utf-8')
        else:
            df_with = pd.read_excel(filepath)

        metrics_with = compare_markups(
            df_gold, df_with,
            markup_slot=WITH_CONTEXT_SLOT
        )

        all_results[context_type] = {
            field_key: data['metrics'] if data['metrics'] else {}
            for field_key, data in metrics_with.items()
        }
        all_results[f'{context_type}_samples'] = {
            field_key: data['samples']
            for field_key, data in metrics_with.items()
        }

        # Вывод метрик
        for field_key, metrics in all_results[context_type].items():
            if metrics:
                print(f'  {field_key}: F1={metrics["f1"]:.4f}, F2={metrics["f2"]:.4f}, '
                      f'Precision={metrics["precision"]:.4f}, Recall={metrics["recall"]:.4f}, '
                      f'Kappa={metrics["kappa"]:.4f}')
            else:
                print(f'  {field_key}: нет данных')

    return all_results


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_metrics_comparison(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    context_type: str
):
    """
    Создание и сохранение графиков метрик для одного типа контекста.

    Сравнивает: без контекста vs с контекстом (указанным типом)

    Структура all_results:
    - all_results['without_context'][field_key][metric_key] = value
    - all_results[context_type][field_key][metric_key] = value
    """
    os.makedirs(output_dir, exist_ok=True)

    field_names = {
        'economic_narrative': 'Economic Narrative',
        'narrative_strength': 'Narrative Strength',
        'economic_effect': 'Economic Effect',
        'information_resonance': 'Information Resonance'
    }

    metric_names = {
        'f1': 'F1 Score',
        'f2': 'F2 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'kappa': "Cohen's Kappa"
    }

    # Colors for two markup types
    colors = {
        'without_context': '#A23B72',  # Dark pink
        'with_context': '#2E86AB'      # Blue
    }

    labels = {
        'without_context': 'Without Context',
        'with_context': f'With Context ({context_type})'
    }

    # Получаем список полей из данных контекста
    context_data = all_results.get(context_type, {})
    fields = list(context_data.keys()) if context_data else []

    if not fields:
        print(f'  Нет данных для визуализации {context_type}')
        return

    # 4 отдельных графика — по одному для каждой метрики
    for metric_key, metric_name in metric_names.items():
        plt.figure(figsize=(12, 6))

        x = np.arange(len(fields))
        width = 0.35

        values_without = []
        values_with = []

        for f in fields:
            m_without = all_results.get('without_context', {}).get(f, {})
            m_with = context_data.get(f, {})
            values_without.append(m_without.get(metric_key, 0) if m_without else 0)
            values_with.append(m_with.get(metric_key, 0) if m_with else 0)

        bars1 = plt.bar(x - width/2, values_without, width,
                       color=colors['without_context'], edgecolor='black', linewidth=1.2,
                       label=labels['without_context'])
        bars2 = plt.bar(x + width/2, values_with, width,
                       color=colors['with_context'], edgecolor='black', linewidth=1.2,
                       label=labels['with_context'])

        # Добавляем значения на столбцы
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.ylim(0, 1.1)
        plt.title(f'{metric_name}', fontsize=18, fontweight='bold')
        plt.ylabel('Value', fontsize=16)
        plt.xticks(x, [field_names[f] for f in fields], rotation=25, ha='right', fontsize=14)
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        filename = os.path.join(output_dir, f'metrics_{metric_key}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Сохранено: {filename}')

    # 1 совмещенный график — все метрики вместе
    plt.figure(figsize=(16, 8))

    x = np.arange(len(fields))
    width = 0.20

    for i, (metric_key, metric_name) in enumerate(metric_names.items()):
        values_without = [all_results.get('without_context', {}).get(f, {}).get(metric_key, 0) for f in fields]
        values_with = [all_results.get(context_type, {}).get(f, {}).get(metric_key, 0) for f in fields]

        # Группируем по 2 столбца на поле (без контекста и с контекстом)
        offset_base = x + (i - 1.5) * width
        plt.bar(offset_base - width/2, values_without, width,
               label=f'{metric_name} (без контекста)', color=colors['without_context'],
               edgecolor='black', linewidth=1, alpha=0.8)
        plt.bar(offset_base + width/2, values_with, width,
               label=f'{metric_name} ({context_type})', color=colors['with_context'],
               edgecolor='black', linewidth=1, alpha=0.8)

    plt.xlabel('Field', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.title(f'Metrics Comparison: Without Context vs With Context ({context_type})', fontsize=18, fontweight='bold')
    plt.xticks(x, [field_names[f] for f in fields], rotation=25, ha='right', fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    filename = os.path.join(output_dir, 'metrics_combined.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Сохранено: {filename}')


def plot_all_contexts_comparison(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str
):
    """
    Создание сравнительного графика всех типов контекста вместе.
    """
    os.makedirs(output_dir, exist_ok=True)

    field_names = {
        'economic_narrative': 'Economic Narrative',
        'narrative_strength': 'Narrative Strength',
        'economic_effect': 'Economic Effect',
        'information_resonance': 'Information Resonance'
    }

    metric_names = {
        'f1': 'F1 Score',
        'f2': 'F2 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'kappa': "Cohen's Kappa"
    }

    context_types = ['without_context'] + CONTEXT_TYPES
    context_labels = {
        'without_context': 'Without Context',
        'llm_context': 'llm_context',
        'llm_context_features': 'llm_context_features',
        'llm_context_features_full': 'llm_context_features_full'
    }

    # Цвета для каждого типа контекста
    colors = {
        'without_context': '#A23B72',
        'llm_context': '#2E86AB',
        'llm_context_features': '#28B463',
        'llm_context_features_full': '#D68910'
    }

    fields = list(all_results.get('llm_context', {}).keys()) if 'llm_context' in all_results else []
    if not fields:
        print('  Нет данных для сводного графика')
        return

    # Для каждой метрики создаём отдельный график
    for metric_key, metric_name in metric_names.items():
        plt.figure(figsize=(14, 7))

        x = np.arange(len(fields))
        width = 0.8 / len(context_types)

        for i, ctx_type in enumerate(context_types):
            values = [all_results.get(ctx_type, {}).get(f, {}).get(metric_key, 0) for f in fields]
            offset = x - 0.4 + (i + 0.5) * width
            plt.bar(offset, values, width,
                   label=context_labels[ctx_type],
                   color=colors[ctx_type],
                   edgecolor='black', linewidth=1, alpha=0.85)

        plt.xlabel('Field', fontsize=16)
        plt.ylabel('Value', fontsize=16)
        plt.title(f'{metric_name} - All Context Types Comparison', fontsize=18, fontweight='bold')
        plt.xticks(x, [field_names[f] for f in fields], rotation=25, ha='right', fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(loc='upper right', fontsize=14)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        filename = os.path.join(output_dir, f'comparison_{metric_key}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Сохранено: {filename}')

    # Сводный график F1 для всех метрик
    plt.figure(figsize=(14, 7))

    x = np.arange(len(fields))
    width = 0.8 / len(context_types)

    for i, ctx_type in enumerate(context_types):
        values = [all_results.get(ctx_type, {}).get(f, {}).get('f1', 0) for f in fields]
        offset = x - 0.4 + (i + 0.5) * width
        plt.bar(offset, values, width,
               label=context_labels[ctx_type],
               color=colors[ctx_type],
               edgecolor='black', linewidth=1, alpha=0.85)

    plt.xlabel('Field', fontsize=16)
    plt.ylabel('F1 Score', fontsize=16)
    plt.title('F1 Score - All Context Types Comparison', fontsize=18, fontweight='bold')
    plt.xticks(x, [field_names[f] for f in fields], rotation=25, ha='right', fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    filename = os.path.join(output_dir, 'comparison_f1_all.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Сохранено: {filename}')


# ==================== СОХРАНЕНИЕ МЕТРИК В XLSX ====================

def save_metrics_xlsx(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str
):
    """Сохранение метрик в XLSX файл."""
    data = []
    fields = ['economic_narrative', 'narrative_strength', 'economic_effect', 'information_resonance']
    metrics_list = ['f1', 'f2', 'precision', 'recall', 'kappa']

    for field_key in fields:
        row = {'Field': field_key}

        # Метрики без контекста
        m_without = all_results.get('without_context', {}).get(field_key, {})
        for m in metrics_list:
            row[f'{m.upper()} (without context)'] = m_without.get(m, 0) if m_without else 0
        row['Samples (without context)'] = all_results.get('without_context_samples', {}).get(field_key, 0)

        # Метрики для каждого типа контекста
        for ctx_type in CONTEXT_TYPES:
            m_with = all_results.get(ctx_type, {}).get(field_key, {})
            for m in metrics_list:
                row[f'{m.upper()} ({ctx_type})'] = m_with.get(m, 0) if m_with else 0
            row[f'Samples ({ctx_type})'] = all_results.get(f'{ctx_type}_samples', {}).get(field_key, 0)

        data.append(row)

    df_metrics = pd.DataFrame(data)
    df_metrics.to_excel(output_path, index=False)
    print(f'  Сохранено: {output_path}')


def save_summary_xlsx(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str
):
    """Сохранение сводной таблицы с лучшими метриками."""
    data = []
    fields = ['economic_narrative', 'narrative_strength', 'economic_effect', 'information_resonance']
    metrics_list = ['f1', 'f2', 'precision', 'recall', 'kappa']

    for field_key in fields:
        row = {'Field': field_key}

        for metric in metrics_list:
            # Find best value among all context types
            best_value = 0
            best_context = 'without_context'

            m_without = all_results.get('without_context', {}).get(field_key, {})
            if m_without:
                best_value = m_without.get(metric, 0)

            for ctx_type in CONTEXT_TYPES:
                m_with = all_results.get(ctx_type, {}).get(field_key, {})
                if m_with:
                    value = m_with.get(metric, 0)
                    if value > best_value:
                        best_value = value
                        best_context = ctx_type

            row[f'{metric.upper()}'] = best_value
            row[f'{metric.upper()}_best'] = best_context

        data.append(row)

    df_summary = pd.DataFrame(data)
    df_summary.to_excel(output_path, index=False)
    print(f'  Сохранено: {output_path}')


# ==================== ПОИСК ПОСЛЕДНЕГО ФАЙЛА ====================

def find_latest_markup(base_dir: str, context_type: str) -> Optional[str]:
    """
    Поиск последнего файла с разметкой для указанного типа контекста.

    Использует точное совпадение имени файла: после context_type должен идти timestamp.
    """
    # Шаблон: goldenset_remarked_<context_type>_YYYYMMDD_HHMMSS.xlsx
    pattern = f'goldenset_remarked_{context_type}_*'
    full_pattern = os.path.join(base_dir, pattern)
    files = glob.glob(full_pattern)

    # Фильтруем только точные совпадения (чтобы llm_context не включал llm_context_features)
    exact_files = []
    for f in files:
        basename = os.path.basename(f)
        # Проверяем что имя файла начинается с точного context_type за которым следует timestamp
        expected_prefix = f'goldenset_remarked_{context_type}_'
        if basename.startswith(expected_prefix):
            # После префикса должно быть YYYYMMDD_HHMMSS (15 символов) + расширение
            rest = basename[len(expected_prefix):]
            # Проверяем формат: 8 цифр, _, 6 цифр, .
            if len(rest) >= 16 and rest[:8].isdigit() and rest[8] == '_' and rest[9:15].isdigit():
                exact_files.append(f)

    if not exact_files:
        return None

    # Сортируем по имени файла (включая timestamp) и берём последний
    return sorted(exact_files, reverse=True)[0]


def find_context_files(base_dir: str) -> Dict[str, str]:
    """Поиск файлов разметки для каждого типа контекста."""
    context_files = {}

    for ctx_type in CONTEXT_TYPES:
        filepath = find_latest_markup(base_dir, ctx_type)

        if filepath:
            context_files[ctx_type] = filepath
            print(f'  {ctx_type}: {os.path.basename(filepath)}')
        else:
            print(f'  {ctx_type}: не найдено')

    return context_files


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Вычисление метрик сравнения разметок с разными типами контекста'
    )
    parser.add_argument(
        '--goldenset',
        type=str,
        default=GOLDENSET_CSV,
        help=f'Файл золотого стандарта (по умолчанию: {GOLDENSET_CSV})'
    )
    parser.add_argument(
        '--without-context',
        type=str,
        default=WITHOUT_CONTEXT_XLSX,
        help=f'Файл разметки без контекста (по умолчанию: {WITHOUT_CONTEXT_XLSX})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='metrics_plots',
        help='Базовая директория для графиков (по умолчанию: metrics_plots)'
    )
    args = parser.parse_args()

    print('=' * 60)
    print('Вычисление метрик сравнения разметок')
    print('Обычная (без контекста) vs Обогащённая (3 типа контекста)')
    print('=' * 60)

    # Загрузка золотого стандарта
    goldenset_path = os.path.join(_base_dir, args.goldenset)
    print(f'\nЗагрузка золотого стандарта из: {goldenset_path}')
    df_gold = pd.read_csv(goldenset_path, encoding='utf-8')
    print(f'  Строк: {len(df_gold)}')

    # Загрузка обычной разметки (без контекста)
    without_context_path = os.path.join(_base_dir, args.without_context)
    print(f'\nЗагрузка обычной разметки (без контекста) из: {without_context_path}')
    df_without = pd.read_excel(without_context_path)
    print(f'  Строк: {len(df_without)}')

    # Поиск файлов разметки для каждого типа контекста
    print('\nПоиск файлов обогащённой разметки:')
    context_files = find_context_files(_base_dir)

    if not context_files:
        print('\nНе найдены файлы обогащённой разметки!')
        print('Сначала запустите remarkup_qwen3.py для каждого типа контекста:')
        print('  python remarkup_qwen3.py --context-type llm_context')
        print('  python remarkup_qwen3.py --context-type llm_context_features')
        print('  python remarkup_qwen3.py --context-type llm_context_features_full')
        return

    # Вычисление метрик
    print('\nВычисление метрик:')
    print('-' * 60)
    all_results = compare_all_contexts(df_gold, df_without, context_files)
    print('-' * 60)

    # Визуализация - отдельные папки для каждого типа контекста
    base_output_dir = os.path.join(_base_dir, args.output_dir)

    print(f'\nСоздание графиков (отдельные папки для каждого типа контекста):')
    for ctx_type in CONTEXT_TYPES:
        # Проверяем наличие метрик для этого типа контекста
        if ctx_type in all_results and 'economic_narrative' in all_results.get(ctx_type, {}):
            output_dir = os.path.join(base_output_dir, ctx_type)
            print(f'\n  {ctx_type}:')
            plot_metrics_comparison(all_results, output_dir, ctx_type)
        else:
            print(f'\n  {ctx_type}: нет данных для визуализации')

    # Сводные графики для всех типов контекста
    print(f'\nСоздание сводных графиков (сравнение всех типов):')
    summary_dir = os.path.join(base_output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    plot_all_contexts_comparison(all_results, summary_dir)

    # Сохранение метрик
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_xlsx_path = os.path.join(_base_dir, f'metrics_comparison_{stamp}.xlsx')
    summary_xlsx_path = os.path.join(_base_dir, f'metrics_summary_{stamp}.xlsx')

    print(f'\nСохранение метрик в:')
    save_metrics_xlsx(all_results, metrics_xlsx_path)
    save_summary_xlsx(all_results, summary_xlsx_path)

    print('\n' + '=' * 60)
    print('Завершено!')
    print('=' * 60)
    print(f'\nРезультаты:')
    print(f'  Графики по типам контекста: {base_output_dir}/<context_type>/')
    print(f'  Сводные графики: {summary_dir}/')
    print(f'  Метрики XLSX: {metrics_xlsx_path}')
    print(f'  Сводная таблица XLSX: {summary_xlsx_path}')


if __name__ == '__main__':
    main()
