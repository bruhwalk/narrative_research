"""
Скрипт для автоматического запуска разметки всеми 3 типами контекста
и последующего вычисления метрик.

Использование:
    python run_all_contexts.py [--start-row N] [--max-rows N] [--num-keys N]

Пример:
    python run_all_contexts.py --start-row 0 --max-rows 100 --num-keys 5
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# ==================== КОНФИГУРАЦИЯ ====================

CONTEXT_TYPES = [
    'llm_context',
    'llm_context_features',
    'llm_context_features_full'
]

INPUT_CSV = 'goldenset_with_llm_contexts_final.csv'

# Слоты для каждой разметки (можно использовать один слот т.к. файлы разные)
SLOT = 4

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Запуск разметки всеми 3 типами контекста'
    )
    parser.add_argument(
        '--start-row',
        type=int,
        default=0,
        help='Начальная строка (по умолчанию: 0)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=100,
        help='Максимальное количество строк для теста, 0 = все (по умолчанию: 100)'
    )
    parser.add_argument(
        '--num-keys',
        type=int,
        default=5,
        help='Количество API ключей для параллелизации (по умолчанию: 5)'
    )
    parser.add_argument(
        '--input-csv',
        type=str,
        default=INPUT_CSV,
        help=f'Входной CSV файл (по умолчанию: {INPUT_CSV})'
    )
    parser.add_argument(
        '--skip-markup',
        action='store_true',
        help='Пропустить разметку, только вычисление метрик'
    )
    parser.add_argument(
        '--skip-metrics',
        action='store_true',
        help='Пропустить вычисление метрик'
    )
    args = parser.parse_args()

    print('=' * 70)
    print('Автоматический запуск разметки всеми типами контекста')
    print('=' * 70)
    print(f'\nПараметры:')
    print(f'  Входной файл: {args.input_csv}')
    print(f'  Начальная строка: {args.start_row}')
    print(f'  Максимум строк: {args.max_rows if args.max_rows > 0 else "все"}')
    print(f'  Количество ключей: {args.num_keys}')
    print(f'  Слот разметки: {SLOT}')

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

    # ==================== ЗАПУСК РАЗМЕТКИ ====================
    if not args.skip_markup:
        print('\n' + '=' * 70)
        print('ЗАПУСК РАЗМЕТКИ')
        print('=' * 70)

        for i, context_type in enumerate(CONTEXT_TYPES, 1):
            print(f'\n[{i}/{3}] Запуск разметки с контекстом: {context_type}')
            print('-' * 70)

            cmd = [
                sys.executable,
                'remarkup_qwen3.py',
                '--context-type', context_type,
                '--slot', str(SLOT),
                '--start-row', str(args.start_row),
                '--max-rows', str(args.max_rows),
                '--num-keys', str(args.num_keys),
                '--input-csv', args.input_csv
            ]

            print(f'Команда: {" ".join(cmd)}')
            print()

            try:
                result = subprocess.run(cmd, cwd=base_dir, check=True)
                print(f'\n[OK] Разметка {context_type} завершена успешно')
            except subprocess.CalledProcessError as e:
                print(f'\n[ERROR] Разметка {context_type} завершилась с ошибкой: {e}')
                print('Продолжение работы...')

    # ==================== ВЫЧИСЛЕНИЕ МЕТРИК ====================
    if not args.skip_metrics:
        print('\n' + '=' * 70)
        print('ВЫЧИСЛЕНИЕ МЕТРИК')
        print('=' * 70)

        cmd = [
            sys.executable,
            'compute_metrics.py'
        ]

        print(f'Команда: {" ".join(cmd)}')
        print()

        try:
            result = subprocess.run(cmd, cwd=base_dir, check=True)
            print('\n[OK] Вычисление метрик завершено успешно')
        except subprocess.CalledProcessError as e:
            print(f'\n[ERROR] Вычисление метрик завершилось с ошибкой: {e}')

    # ==================== ЗАВЕРШЕНИЕ ====================
    print('\n' + '=' * 70)
    print('ЗАВЕРШЕНО')
    print('=' * 70)

    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\nВремя завершения: {stamp}')
    print('\nРезультаты:')
    print(f'  Файлы разметки: goldenset_remarked_<context_type>_*.xlsx')
    print(f'  Графики: metrics_plots/<context_type>/')
    print(f'  Сводные графики: metrics_plots/summary/')
    print(f'  Метрики XLSX: metrics_comparison_*.xlsx')
    print(f'  Сводная таблица: metrics_summary_*.xlsx')

    print('\nPrimary outputs:')
    print('  - metrics_plots/summary/comparison_f1_all.png')
    print('  - metrics_summary_*.xlsx')


if __name__ == '__main__':
    main()
