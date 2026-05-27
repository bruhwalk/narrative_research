"""
LLM as Judge - Аналитика результатов
Создаёт отчёт с графиками и статистикой по всем судьям
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для сохранения без GUI
plt.rcParams['font.family'] = 'DejaVu Sans'  # Поддержка кириллицы


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', BASE_DIR)
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'llm_judge_analytics')
RESULTS_DIR_NO_HUMAN = os.path.join(OUTPUT_DIR, 'llm_judge_analytics_no_human')

# ELO параметры
K_FACTOR = 32
BASE_RATING = 1500

# Маппинг имён классификаторов
CLASSIFIER_NAMES = {
    'human': 'Human',
    'llm_1': 'gpt-oss:20b',
    'llm_2': 'gpt-oss:120b',
    'llm_3': 'qwen3-vl:235b-instruct',
    'llm_4': 'gemma3:27b',
    'llm_5': 'qwen3-vl:8b',
    'llm_6': 'WeDLM-8B-Instruct'
}

# Столбцы с total баллами
TOTAL_COLS = ['human_total', 'llm_1_total', 'llm_2_total', 'llm_3_total', 'llm_4_total', 'llm_5_total', 'llm_6_total']

# Обратный маппинг
NAME_TO_CL = {v: k for k, v in CLASSIFIER_NAMES.items()}

# Классификаторы без человека
LLM_ONLY_CLASSIFIERS = [f'llm_{i}' for i in range(1, 7)]


# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

def find_result_files():
    """Поиск файлов с результатами"""
    pattern = os.path.join(OUTPUT_DIR, 'judge_results_*.xlsx')
    files = glob.glob(pattern)
    # Исключаем бэкапы и аналитику
    files = [f for f in files if 'backup' not in f and 'analytics' not in f]
    return sorted(files)


def load_judge_results(files):
    """Загрузка результатов всех судей"""
    all_data = {}
    
    for f in files:
        filename = os.path.basename(f)
        # Извлекаем имя судьи из имени файла
        match = re.search(r'judge_results_(.+)\.xlsx', filename)
        if match:
            judge_name = match.group(1).replace('_', ' ').replace('.', ':')
            # Исправляем имена моделей
            judge_name = judge_name.replace('minimax-m2 5', 'minimax-m2.5')
            judge_name = judge_name.replace('deepseek-v3 2', 'deepseek-v3.2')
            judge_name = judge_name.replace('qwen3 5', 'qwen3.5')
        else:
            judge_name = filename.replace('.xlsx', '')
        
        try:
            df = pd.read_excel(f, sheet_name='Результаты')
            all_data[judge_name] = df
            print(f"Загружен {judge_name}: {len(df)} строк")
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")
    
    return all_data


# ============================================================================
# ELO РАСЧЁТ
# ============================================================================

class ELOCalculator:
    """Калькулятор ELO рейтинга"""
    
    def __init__(self, k_factor=K_FACTOR, base_rating=BASE_RATING, include_human=True):
        self.k = k_factor
        self.base = base_rating
        self.ratings = {}
        self.wins = defaultdict(int)
        self.losses = defaultdict(int)
        self.matches = defaultdict(int)
        self.include_human = include_human
    
    def calculate(self, df):
        """
        Расчёт ELO по данным судьи
        Победитель определяется по максимальному total значению в строке
        Пропущенные оценки (NaN) не учитываются
        """
        if self.include_human:
            classifiers = ['human'] + [f'llm_{i}' for i in range(1, 7)]
        else:
            classifiers = [f'llm_{i}' for i in range(1, 7)]
        
        # Инициализация
        for cl in classifiers:
            self.ratings[cl] = self.base
        
        # Обработка каждой строки
        for idx, row in df.iterrows():
            # Находим победителя по максимальному total
            best = None
            best_score = -1
            
            for cl in classifiers:
                col = f'{cl}_total'
                if col in row.index:
                    score = row[col]
                    if pd.notna(score) and score > best_score:
                        best_score = score
                        best = cl
            
            if not best or best not in classifiers:
                continue  # Пропущенная оценка
            
            # Победитель получает +K * (1 - expected)
            # Проигравшие получают +K * (0 - expected)
            
            winner = best
            self.wins[winner] += 1
            self.matches[winner] += 1
            
            for loser in classifiers:
                if loser == winner:
                    continue
                
                # Ожидаемый результат для победителя против каждого проигравшего
                expected_winner = 1 / (1 + 10 ** ((self.ratings[loser] - self.ratings[winner]) / 400))
                
                # Обновление рейтинга победителя
                self.ratings[winner] += self.k * (1 - expected_winner)
                
                # Ожидаемый результат для проигравшего
                expected_loser = 1 / (1 + 10 ** ((self.ratings[winner] - self.ratings[loser]) / 400))
                
                # Обновление рейтинга проигравшего
                self.ratings[loser] += self.k * (0 - expected_loser)
                self.losses[loser] += 1
                self.matches[loser] += 1
        
        return self.get_standings()
    
    def get_standings(self):
        """Таблица лидеров"""
        standings = []
        for cl in self.ratings:
            w = self.wins[cl]
            l = self.losses[cl]
            m = self.matches[cl]
            standings.append({
                'Классификатор': CLASSIFIER_NAMES.get(cl, cl),
                'ELO': round(self.ratings[cl], 1),
                'Побед': w,
                'Поражений': l,
                'Матчей': m,
                'Win Rate %': round(w / max(1, m) * 100, 1) if m > 0 else 0
            })
        return sorted(standings, key=lambda x: x['ELO'], reverse=True)


# ============================================================================
# АНАЛИЗ ПРИЧИН
# ============================================================================

def analyze_best_reasons(df):
    """Анализ причин победы из best_reason"""
    reasons = df['best_reason'].dropna().tolist()
    
    # Ключевые слова для анализа
    keywords = {
        'нарратив': ['нарратив', 'narrative', 'да', 'нет'],
        'сила': ['сил', 'strength', 'мощн'],
        'эффект': ['эффект', 'effect', 'влияни'],
        'резонанс': ['резонанс', 'resonance', 'отклик'],
        'тема': ['тем', 'topic', 'соответству'],
        'точный': ['точн', 'accurat', 'верн', 'правильн'],
        'лучший': ['лучш', 'best', 'наиболее'],
        'ошибка': ['ошиб', 'error', 'неверн', 'wrong'],
    }
    
    analysis = {k: 0 for k in keywords.keys()}
    total = len(reasons)
    
    for reason in reasons:
        reason_lower = reason.lower()
        for key, words in keywords.items():
            if any(word in reason_lower for word in words):
                analysis[key] += 1
    
    # Проценты
    percentages = {k: round(v / max(1, total) * 100, 1) for k, v in analysis.items()}
    
    return analysis, percentages, total


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def create_elo_comparison_plot(standings_dict, output_path, include_human=True):
    """Elo rating comparison across judge models"""

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('ELO Rating Comparison Across Judge Models', fontsize=14, fontweight='bold')

    if include_human:
        classifiers = list(CLASSIFIER_NAMES.values())
    else:
        classifiers = [CLASSIFIER_NAMES[cl] for cl in LLM_ONLY_CLASSIFIERS]

    x = np.arange(len(classifiers))
    width = 0.25

    judges = list(standings_dict.keys())
    # High contrast colors for different judges
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    for i, (judge, standings) in enumerate(standings_dict.items()):
        elo_vals = []
        for cl_name in classifiers:
            for s in standings:
                if s['Классификатор'] == cl_name:
                    elo_vals.append(s['ELO'])
                    break
            else:
                elo_vals.append(0)

        ax.bar(x + i * width, elo_vals, width, label=judge[:20], color=colors[i % len(colors)], edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
    ax.set_title('ELO Rating by Judge Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * (len(judges) - 1) / 2)
    ax.set_xticklabels([c[:15] for c in classifiers], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ELO comparison plot: {output_path}")


def create_winrate_comparison_plot(standings_dict, output_path, include_human=True):
    """Win rate comparison across models"""

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Win Rate Comparison Across Models', fontsize=14, fontweight='bold')

    if include_human:
        classifiers = list(CLASSIFIER_NAMES.values())
    else:
        classifiers = [CLASSIFIER_NAMES[cl] for cl in LLM_ONLY_CLASSIFIERS]

    x = np.arange(len(classifiers))
    width = 0.25

    judges = list(standings_dict.keys())
    # High contrast colors for different judges
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    for i, (judge, standings) in enumerate(standings_dict.items()):
        wr_vals = []
        for cl_name in classifiers:
            for s in standings:
                if s['Классификатор'] == cl_name:
                    wr_vals.append(s['Win Rate %'])
                    break
            else:
                wr_vals.append(0)

        ax.bar(x + i * width, wr_vals, width, label=judge[:20], color=colors[i % len(colors)], edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate by Judge Model', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * (len(judges) - 1) / 2)
    ax.set_xticklabels([c[:15] for c in classifiers], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Win Rate comparison plot: {output_path}")

def create_bar_plot(standings_dict, output_path, include_human=True):
    """Сравнительный bar plot для всех судей"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнение классификаторов по судьям', fontsize=14, fontweight='bold')

    # High contrast colors for classifiers
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    if include_human:
        classifiers = list(CLASSIFIER_NAMES.values())
    else:
        classifiers = [CLASSIFIER_NAMES[cl] for cl in LLM_ONLY_CLASSIFIERS]

    # 1. ELO рейтинг
    ax1 = axes[0, 0]
    x = np.arange(len(classifiers))
    width = 0.25

    judges = list(standings_dict.keys())
    judge_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for i, (judge, standings) in enumerate(standings_dict.items()):
        elo_vals = []
        for cl_name in classifiers:
            for s in standings:
                if s['Классификатор'] == cl_name:
                    elo_vals.append(s['ELO'])
                    break
            else:
                elo_vals.append(0)

        ax1.bar(x + i * width, elo_vals, width, label=judge[:15], color=judge_colors[i % len(judge_colors)], edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Классификатор', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ELO рейтинг', fontsize=11, fontweight='bold')
    ax1.set_title('ELO по судьям', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([c[:12] for c in classifiers], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax1.set_axisbelow(True)

    # 2. Победы (абсолютные)
    ax2 = axes[0, 1]
    for i, (judge, standings) in enumerate(standings_dict.items()):
        wins = [s['Побед'] for s in standings]
        ordered_wins = []
        for cl_name in classifiers:
            for s in standings:
                if s['Классификатор'] == cl_name:
                    ordered_wins.append(s['Побед'])
                    break
            else:
                ordered_wins.append(0)

        ax2.bar(x + i * width, ordered_wins, width, label=judge[:15], color=judge_colors[i % len(judge_colors)], edgecolor='black', linewidth=1.2)

    ax2.set_xlabel('Классификатор', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Побед', fontsize=11, fontweight='bold')
    ax2.set_title('Победы по судьям', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([c[:12] for c in classifiers], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax2.set_axisbelow(True)

    # 3. Win Rate %
    ax3 = axes[1, 0]
    for i, (judge, standings) in enumerate(standings_dict.items()):
        wr = [s['Win Rate %'] for s in standings]
        ordered_wr = []
        for cl_name in classifiers:
            for s in standings:
                if s['Классификатор'] == cl_name:
                    ordered_wr.append(s['Win Rate %'])
                    break
            else:
                ordered_wr.append(0)

        ax3.bar(x + i * width, ordered_wr, width, label=judge[:15], color=judge_colors[i % len(judge_colors)], edgecolor='black', linewidth=1.2)

    ax3.set_xlabel('Классификатор', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Win Rate %', fontsize=11, fontweight='bold')
    ax3.set_title('Процент побед по судьям', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([c[:12] for c in classifiers], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax3.set_axisbelow(True)

    # 4. Сводный ELO (средний по судьям)
    ax4 = axes[1, 1]
    avg_elo = []
    for cl in classifiers:
        elo_sum = 0
        count = 0
        for standings in standings_dict.values():
            for s in standings:
                if s['Классификатор'] == cl:
                    elo_sum += s['ELO']
                    count += 1
                    break
        avg_elo.append(elo_sum / max(1, count))

    bars = ax4.bar(classifiers, avg_elo, color=colors[:len(classifiers)], edgecolor='black', linewidth=1.2)
    ax4.set_xlabel('Классификатор', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Средний ELO', fontsize=11, fontweight='bold')
    ax4.set_title('Средний ELO по всем судьям', fontsize=12, fontweight='bold')
    ax4.set_xticklabels([c[:12] for c in classifiers], rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.7)
    ax4.set_axisbelow(True)

    # Подписи значений
    for bar, val in zip(bars, avg_elo):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранён график: {output_path}")


def create_judge_specific_plot(standings, judge_name, output_path, include_human=True):
    """График для конкретного судьи"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Результаты судьи: {judge_name}', fontsize=14, fontweight='bold')

    # Фильтруем human если нужно
    if include_human:
        filtered_standings = standings
    else:
        filtered_standings = [s for s in standings if s['Классификатор'] != 'Human']

    classifiers = [s['Классификатор'] for s in filtered_standings]
    elo_vals = [s['ELO'] for s in filtered_standings]
    wins = [s['Побед'] for s in filtered_standings]
    wr = [s['Win Rate %'] for s in filtered_standings]

    # High contrast colors
    colors = ['#FF6B6B' if c == 'Human' else '#4ECDC4' for c in classifiers]

    # ELO
    ax1 = axes[0]
    bars1 = ax1.barh(classifiers, elo_vals, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('ELO рейтинг', fontsize=11, fontweight='bold')
    ax1.set_title('ELO рейтинг', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.7)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, elo_vals):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=10, fontweight='bold')

    # Победы
    ax2 = axes[1]
    bars2 = ax2.barh(classifiers, wins, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Побед', fontsize=11, fontweight='bold')
    ax2.set_title('Абсолютные победы', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.7)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, wins):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=10, fontweight='bold')

    # Win Rate
    ax3 = axes[2]
    bars3 = ax3.barh(classifiers, wr, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Win Rate %', fontsize=11, fontweight='bold')
    ax3.set_title('Процент побед', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.7)
    ax3.set_axisbelow(True)
    for bar, val in zip(bars3, wr):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранён график для {judge_name}: {output_path}")


def create_reason_analysis(reason_data_dict, output_path):
    """Анализ причин победы"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Анализ причин победы (best_reason)', fontsize=14)
    
    # Словарь русских названий
    ru_names = {
        'нарратив': 'Нарратив (Да/Нет)',
        'сила': 'Сила нарратива',
        'эффект': 'Экономический эффект',
        'резонанс': 'Инф. резонанс',
        'тема': 'Тема',
        'точный': 'Точность',
        'лучший': 'Лучший результат',
        'ошибка': 'Ошибки',
    }
    
    # 1. Общая статистика по всем судьям
    ax1 = axes[0]
    all_keywords = list(ru_names.keys())
    all_counts = defaultdict(int)
    
    for judge, (analysis, _, _) in reason_data_dict.items():
        for k, v in analysis.items():
            all_counts[k] += v
    
    counts = [all_counts[k] for k in all_keywords]
    labels = [ru_names[k] for k in all_keywords]
    
    bars = ax1.barh(labels, counts, color='steelblue')
    ax1.set_xlabel('Упоминаний')
    ax1.set_title('Частота упоминания причин (все судьи)')
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=9)
    
    # 2. По судьям (топ причины)
    ax2 = axes[1]
    judges = list(reason_data_dict.keys())
    x = np.arange(len(all_keywords))
    width = 0.25
    
    for i, (judge, (_, percentages, _)) in enumerate(reason_data_dict.items()):
        pcts = [percentages[k] for k in all_keywords]
        ax2.bar(x + i * width, pcts, width, label=judge[:15])
    
    ax2.set_xlabel('Причина')
    ax2.set_ylabel('% упоминаний')
    ax2.set_title('Процент упоминания причин по судьям')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([ru_names[k][:15] for k in all_keywords], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранён анализ причин: {output_path}")


# ============================================================================
# СОЗДАНИЕ ОТЧЁТА
# ============================================================================

def create_analytics_folder():
    """Создание папки для аналитики"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Создана папка: {RESULTS_DIR}")
    return RESULTS_DIR


def create_analytics_folder_no_human():
    """Создание папки для аналитики без человека"""
    if not os.path.exists(RESULTS_DIR_NO_HUMAN):
        os.makedirs(RESULTS_DIR_NO_HUMAN)
        print(f"Создана папка: {RESULTS_DIR_NO_HUMAN}")
    return RESULTS_DIR_NO_HUMAN


def save_statistics(standings_dict, reason_data_dict, output_path):
    """Сохранение сводной статистики"""
    
    # Сборная таблица
    all_stats = []
    
    for judge, standings in standings_dict.items():
        for s in standings:
            all_stats.append({
                'Судья': judge,
                **s
            })
    
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_excel(output_path, index=False)
    print(f"Сохранена статистика: {output_path}")
    
    # Таблица причин
    reasons_data = []
    for judge, (_, percentages, total) in reason_data_dict.items():
        reasons_data.append({
            'Судья': judge,
            'Всего оценок': total,
            **percentages
        })
    
    df_reasons = pd.DataFrame(reasons_data)
    reasons_path = output_path.replace('.xlsx', '_reasons.xlsx')
    df_reasons.to_excel(reasons_path, index=False)
    print(f"Сохранены причины: {reasons_path}")
    
    return df_stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("LLM as Judge - Аналитика результатов")
    print("="*60)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Создание папок
    results_dir = create_analytics_folder()
    results_dir_no_human = create_analytics_folder_no_human()
    
    # Поиск файлов
    files = find_result_files()
    print(f"\nНайдено файлов: {len(files)}")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    if not files:
        print("Нет файлов для анализа!")
        return
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    all_data = load_judge_results(files)
    
    if not all_data:
        print("Не удалось загрузить данные!")
        return
    
    # Расчёт ELO и статистики (С ЧЕЛОВЕКОМ)
    print("\nРасчёт статистики (с человеком)...")
    standings_dict = {}
    elo_calculators = {}
    
    for judge, df in all_data.items():
        calc = ELOCalculator(include_human=True)
        standings = calc.calculate(df)
        standings_dict[judge] = standings
        elo_calculators[judge] = calc
        print(f"\n{judge}:")
        for s in standings[:3]:
            print(f"  {s['Классификатор']}: ELO={s['ELO']}, Побед={s['Побед']}")
    
    # Расчёт ELO и статистики (БЕЗ ЧЕЛОВЕКА)
    print("\nРасчёт статистики (без человека)...")
    standings_dict_no_human = {}
    elo_calculators_no_human = {}
    
    for judge, df in all_data.items():
        calc = ELOCalculator(include_human=False)
        standings = calc.calculate(df)
        standings_dict_no_human[judge] = standings
        elo_calculators_no_human[judge] = calc
        print(f"\n{judge}:")
        for s in standings[:3]:
            print(f"  {s['Классификатор']}: ELO={s['ELO']}, Побед={s['Побед']}")
    
    # Анализ причин
    print("\nАнализ причин победы...")
    reason_data_dict = {}
    for judge, df in all_data.items():
        analysis, percentages, total = analyze_best_reasons(df)
        reason_data_dict[judge] = (analysis, percentages, total)
        print(f"\n{judge} (всего: {total}):")
        for k, v in sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {k}: {v}%")
    
    # Визуализация (С ЧЕЛОВЕКОМ)
    print("\nCreating visualizations (with human)...")
    create_elo_comparison_plot(standings_dict, os.path.join(results_dir, 'elo_comparison.png'), include_human=True)
    create_winrate_comparison_plot(standings_dict, os.path.join(results_dir, 'winrate_comparison.png'), include_human=True)
    create_bar_plot(standings_dict, os.path.join(results_dir, 'comparison.png'), include_human=True)
    for judge, standings in standings_dict.items():
        safe_name = judge.replace(':', '_').replace(' ', '_')
        create_judge_specific_plot(standings, judge, os.path.join(results_dir, f'judge_{safe_name}.png'), include_human=True)
    create_reason_analysis(reason_data_dict, os.path.join(results_dir, 'reasons_analysis.png'))
    save_statistics(standings_dict, reason_data_dict, os.path.join(results_dir, 'summary_statistics.xlsx'))
    
    # Визуализация (БЕЗ ЧЕЛОВЕКА)
    print("\nCreating visualizations (without human)...")
    create_elo_comparison_plot(standings_dict_no_human, os.path.join(results_dir_no_human, 'elo_comparison.png'), include_human=False)
    create_winrate_comparison_plot(standings_dict_no_human, os.path.join(results_dir_no_human, 'winrate_comparison.png'), include_human=False)
    create_bar_plot(standings_dict_no_human, os.path.join(results_dir_no_human, 'comparison.png'), include_human=False)
    for judge, standings in standings_dict_no_human.items():
        safe_name = judge.replace(':', '_').replace(' ', '_')
        create_judge_specific_plot(standings, judge, os.path.join(results_dir_no_human, f'judge_{safe_name}.png'), include_human=False)
    create_reason_analysis(reason_data_dict, os.path.join(results_dir_no_human, 'reasons_analysis.png'))
    save_statistics(standings_dict_no_human, reason_data_dict, os.path.join(results_dir_no_human, 'summary_statistics.xlsx'))
    
    print("\n" + "="*60)
    print("АНАЛИТИКА ГОТОВА!")
    print(f"Папка с человеком: {results_dir}")
    print(f"Папка без человека: {results_dir_no_human}")
    print("="*60)
    
    # Краткая сводка (С ЧЕЛОВЕКОМ)
    print("\nИТОГОВАЯ ТАБЛИЦА ELO (с человеком):")
    classifiers = list(CLASSIFIER_NAMES.values())
    avg_elo = {}
    for cl in classifiers:
        elo_sum = 0
        count = 0
        for standings in standings_dict.values():
            for s in standings:
                if s['Классификатор'] == cl:
                    elo_sum += s['ELO']
                    count += 1
                    break
        if count > 0:
            avg_elo[cl] = elo_sum / count
    
    sorted_elo = sorted(avg_elo.items(), key=lambda x: x[1], reverse=True)
    for i, (cl, elo) in enumerate(sorted_elo, 1):
        marker = '[WIN]' if i == 1 else ''
        print(f"  {i}. {cl:25} ELO: {elo:7.1f} {marker}")
    
    # Краткая сводка (БЕЗ ЧЕЛОВЕКА)
    print("\nИТОГОВАЯ ТАБЛИЦА ELO (без человека):")
    llm_classifiers = [CLASSIFIER_NAMES[cl] for cl in LLM_ONLY_CLASSIFIERS]
    avg_elo_no_human = {}
    for cl in llm_classifiers:
        elo_sum = 0
        count = 0
        for standings in standings_dict_no_human.values():
            for s in standings:
                if s['Классификатор'] == cl:
                    elo_sum += s['ELO']
                    count += 1
                    break
        if count > 0:
            avg_elo_no_human[cl] = elo_sum / count
    
    sorted_elo_no_human = sorted(avg_elo_no_human.items(), key=lambda x: x[1], reverse=True)
    for i, (cl, elo) in enumerate(sorted_elo_no_human, 1):
        marker = '[WIN]' if i == 1 else ''
        print(f"  {i}. {cl:25} ELO: {elo:7.1f} {marker}")


if __name__ == '__main__':
    main()
