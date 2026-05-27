"""
Скрипт для повторной разметки датасета моделью qwen3-vl:235b-instruct-cloud
с использованием 5 ключей Ollama параллельно и учетом контекста.

Поддерживает 3 типа обогащения контекста:
- llm_context - базовый mentions-based контекст
- llm_context_features - только структурные признаки
- llm_context_features_full - объединенный контекст
"""

import os
import json
import re
import time
import threading
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# ==================== КОНФИГУРАЦИЯ ====================

# Входные файлы
INPUT_CSV = 'goldenset_with_llm_contexts_final.csv'
PREVIOUS_MARKUP_XLSX = 'LLm markup without context.xlsx'

# Модель и API
MODEL_NAME = 'qwen3-vl:235b-instruct-cloud'
OLLAMA_CLOUD_HOST = 'https://ollama.com'

# API keys are loaded only from environment variables.
# Never commit provider keys into this script.
OLLAMA_API_KEYS = [
    key for key in (
        os.getenv('OLLAMA_API_KEY_1'),
        os.getenv('OLLAMA_API_KEY_2'),
        os.getenv('OLLAMA_API_KEY_3'),
        os.getenv('OLLAMA_API_KEY_4'),
        os.getenv('OLLAMA_API_KEY_5'),
    )
    if key
]

# Параметры запросов
CLOUD_TIMEOUT_S = 120  # Увеличенный таймаут для большой модели
NUM_PREDICT = 2048     # Больше токенов для развернутых ответов
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 2.0
CLOUD_THROTTLE_S = 0.5  # Задержка между запросами

# Параметры разметки
START_ROW = 0
MAX_ROWS = 0  # 0 = все строки
OUTPUT_SLOT = 4  # Номер слота для новой разметки (после 3 предыдущих)

# Тип контекста для обогащения
CONTEXT_TYPE = 'llm_context'  # 'llm_context', 'llm_context_features', 'llm_context_features_full'

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


def _normalize_ollama_model_name(model: str, host: Optional[str]) -> str:
    """Нормализация имени модели для cloud API."""
    if host and model.endswith('-cloud'):
        return model[:-6]
    return model


def _ollama_message_content(resp: Any) -> str:
    """Извлечение текста из ответа Ollama."""
    if resp is None:
        return ''

    if isinstance(resp, dict):
        msg = resp.get('message') or {}
        if isinstance(msg, dict):
            return _strip(msg.get('content'))

    if hasattr(resp, 'message') and hasattr(resp.message, 'content'):
        return _strip(resp.message.content)

    try:
        msg = resp['message']
        return _strip(msg['content'])
    except Exception:
        return _strip(str(resp))


# ==================== ПРОМПТ ====================

def build_prompt(message: str, topic: str, llm_context: str = '', context_type: str = 'llm_context') -> str:
    """
    Построение промпта с учетом контекста из других постов.

    Args:
        message: Текст новости
        topic: Тема новости
        llm_context: Контекст в зависимости от типа
        context_type: Тип контекста ('llm_context', 'llm_context_features', 'llm_context_features_full')
    """
    context_block = ''
    has_context = llm_context and _strip(llm_context)
    
    if has_context:
        if context_type == 'llm_context':
            # Базовый mentions-based контекст
            context_block = (
                '\n\n'
                '=== КОНТЕКСТ (другие посты по этой теме за последние 30 дней) ===\n'
                f'{llm_context}\n'
                '=== КОНЕЦ КОНТЕКСТА ===\n\n'
            )
        elif context_type == 'llm_context_features':
            # Только структурные признаки
            context_block = (
                '\n\n'
                '=== СТАТИСТИКА РАСПРОСТРАНЕНИЯ ТЕМЫ ===\n'
                f'{llm_context}\n'
                '=== КОНЕЦ СТАТИСТИКИ ===\n\n'
            )
        elif context_type == 'llm_context_features_full':
            # Объединенный контекст
            context_block = (
                '\n\n'
                '=== КОНТЕКСТ И СТАТИСТИКА РАСПРОСТРАНЕНИЯ ===\n'
                f'{llm_context}\n'
                '=== КОНЕЦ КОНТЕКСТА ===\n\n'
            )

    return (
        'Ты — эксперт по экономическим новостям и общественному восприятию в России. '
        'Твоя задача — определить, является ли короткая новость экономическим нарративом для широкой российской аудитории.\n\n'
        'Критерии нарратива в вашем понимании:\n\n'
        '1. Релевантность для России: Новость должна иметь прямой или косвенный экономический эффект для жителей России. '
        'Мировые новости без последствий для России — не нарратив.\n\n'
        '2. Широкий общественный резонанс: Фокус на обычных гражданах, а не на узких группах. '
        'Нарратив — это яркая новость, которая может вызвать сильный отклик в массах, желание делиться ею '
        'и влиять на импульсивные решения (например, срочные покупки, вывод средств).\n\n'
        '3. Яркость и сила события: В основе нарратива лежит сильное событие (резкий рост цен, важное политическое заявление, '
        'масштабные санкции), а не рутинная информация.\n\n'
        '4. ВРЕМЕННАЯ ДИНАМИКА (по Шиллеру): Нарратив обладает вирусообразной структурой распределения по времени. '
        'Учитывай предоставленный контекст других постов по этой теме — если новость активно обсуждается в течение '
        'последних 30 дней, это усиливает её нарративный потенциал.\n\n'
        'Проанализируй новость по следующему плану (рассуждения держи в уме, в ответ не выноси):\n\n'
        '[1] Триггер и релевантность:\n'
        'В чём суть новости? Есть ли чёткое триггерное событие (заявление, решение, кризис)?\n'
        'Имеет ли это событие прямые последствия для экономического положения или настроений широких слоёв населения России?\n\n'
        '[2] Эмоциональный заряд и упрощение:\n'
        'Какие эмоции может вызвать текст у обычного человека? (тревога, страх, гнев, оптимизм).\n'
        'Сводится ли основная мысль к простым, обобщающим формулировкам? (Например: «Цены на всё вырастут», «Рубль обвалится», «Наступит дефицит»).\n\n'
        '[3] Логика воздействия и аудитория:\n'
        'Пытается ли новость объяснить, как именно это событие повлияет на жизнь людей, а не просто констатирует факт?\n'
        'Направлена ли новость на массовую аудиторию, а не на профессионалов?\n\n'
        '[4] Источники и резонансный потенциал:\n'
        'Кто является источником или героем новости? (Правительство, ЦБ, известный политик, эксперты в СМИ). '
        'Усиливает ли это авторитетность и потенциальное распространение?\n'
        'Может ли эта новость стать «вирусной» историей для обсуждения в соцсетях и бытовых разговорах?\n\n'
        '[5] Временная динамика и контекст:\n'
        'Как новость соотносится с другими постами по этой теме? Наблюдается ли рост обсуждений?\n'
        'Поддерживает ли контекст вирусный потенциал этой новости?\n\n'
        'Пояснения к полям:\n\n'
        'economic_narrative (Да/Нет): Итоговое решение. Да — если есть триггерное событие, релевантное для широкой российской аудитории, '
        'и новость обладает потенциалом вызвать эмоциональный отклик и массовое обсуждение.\n\n'
        'narrative_strength (1-3): Сила нарративных свойств. Оценивай, насколько текст эмоционален, упрощён и побуждает к действию. '
        '1 — констатация факта; 3 — прямое предупреждение о катастрофических последствиях для всех. '
        'Не нарративные новости в большинстве своем должны получать оценку 1, при этом нарративы тоже могут изредка получать такую оценку.\n\n'
        'economic_effect (-2..2): Влияние на экономическое положение обычного человека в России. Примеры: крах банков: -2, рост в отдельном регионе: -1, отмена санкций: 2.\n\n'
        'topic_agreement (1-3): Твоя оценка правильности выбранной темы (topic). Считай, что тема выбрана из фиксированного списка.\n\n'
        'information_resonance (1-3): Потенциал широкого и эмоционального восприятия в обществе. Высокий резонанс — темы, затрагивающие каждого '
        '(цены на еду, бензин, рубль, важные политические решения для всего населения).\n\n'
        'ВАЖНО: Ответь строго одним JSON-объектом БЕЗ пояснений, БЕЗ markdown и БЕЗ текста вокруг.\n'
        'Схема JSON (ключи строго такие):\n'
        '{\n'
        '  "economic_effect": -2,\n'
        '  "information_resonance": 1,\n'
        '  "topic_agreement": 1,\n'
        '  "economic_narrative": "Да",\n'
        '  "narrative_strength": 1,\n'
        '  "comment": "..."\n'
        '}\n\n'
        f'Тема (topic): {topic}\n'
        f'Новость (message): {message}\n'
        f'{context_block}'
    )


# ==================== OLLAMA CLIENT ====================

def _get_api_key(key_index: int) -> str:
    """Получение API ключа по индексу."""
    if key_index < len(OLLAMA_API_KEYS):
        key = OLLAMA_API_KEYS[key_index]
        if key and key != 'ВАШ_КЛЮЧ_1' and key != 'ВАШ_КЛЮЧ_2' and key != 'ВАШ_КЛЮЧ_3':
            return key
    raise ValueError(f'API ключ {key_index + 1} не задан')


def _ollama_client(host: str, key_index: int = 0):
    """Создание клиента Ollama с указанным API ключом."""
    from ollama import Client

    api_key = _get_api_key(key_index)
    headers = {'Authorization': f'Bearer {api_key}'}
    timeout = CLOUD_TIMEOUT_S

    return Client(host=host, headers=headers, timeout=timeout)


def _should_retry_exc(e: Exception) -> bool:
    """Проверка, стоит ли повторять запрос при ошибке."""
    status = getattr(e, 'status_code', None)
    if isinstance(status, int) and status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    msg = str(e).lower()
    if 'empty model response' in msg:
        return True
    if 'timed out' in msg or 'timeout' in msg:
        return True
    if 'connection' in msg or 'network' in msg:
        return True
    if 'rate limit' in msg:
        return True

    return False


def infer_ollama(model: str, prompt: str, host: str, key_index: int = 0) -> str:
    """Вызов модели Ollama с повторами при ошибках."""
    last_err: Optional[Exception] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            client = _ollama_client(host, key_index)
            resp = client.chat(
                model=_normalize_ollama_model_name(model, host),
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.2, 'num_predict': NUM_PREDICT},
            )

            text = _ollama_message_content(resp)
            if not text:
                raise ValueError('Empty model response')
            return text

        except Exception as e:
            last_err = e
            if attempt < RETRY_ATTEMPTS and _should_retry_exc(e):
                wait_time = RETRY_BACKOFF_S * attempt
                print(f'  Попытка {attempt} не удалась: {type(e).__name__}. Ожидание {wait_time}s...')
                time.sleep(wait_time)
                continue
            break

    raise last_err or ValueError('Empty model response')


# ==================== PARSE JSON ====================

def _json_from_text(text: str) -> Dict[str, Any]:
    """Извлечение JSON из текста ответа."""
    text = _strip(text)
    if not text:
        raise ValueError('Empty text')

    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError('JSON root is not an object')
        return obj
    except Exception:
        pass

    m = re.search(r'\\{.*\\}', text, flags=re.DOTALL)
    if not m:
        raise ValueError('No JSON object found in model output')

    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError('JSON root is not an object')
    return obj


def _to_int(v: Any) -> int:
    """Преобразование значения в целое число."""
    if isinstance(v, bool):
        raise ValueError('Boolean is not allowed')
    if isinstance(v, (int, float)):
        return int(v)
    s = _strip(v)
    if not s:
        raise ValueError('Empty number')
    return int(float(s))


def _validate_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Валидация структуры ответа модели."""
    required = {
        'economic_effect',
        'information_resonance',
        'topic_agreement',
        'economic_narrative',
        'narrative_strength',
        'comment',
    }
    missing = sorted(required - set(obj.keys()))
    if missing:
        raise ValueError('Missing keys: ' + ', '.join(missing))

    payload = {
        'economic_effect': _to_int(obj.get('economic_effect')),
        'information_resonance': _to_int(obj.get('information_resonance')),
        'topic_agreement': _to_int(obj.get('topic_agreement')),
        'narrative_strength': _to_int(obj.get('narrative_strength')),
        'economic_narrative': _strip(obj.get('economic_narrative')),
        'comment': _strip(obj.get('comment')),
    }

    if payload['economic_narrative'] not in {'Да', 'Нет'}:
        raise ValueError('economic_narrative must be \"Да\" or \"Нет\"')

    if payload['economic_effect'] < -2 or payload['economic_effect'] > 2:
        raise ValueError('economic_effect out of range (-2..2)')

    for k in ['information_resonance', 'topic_agreement', 'narrative_strength']:
        if payload[k] < 1 or payload[k] > 3:
            raise ValueError(f'{k} out of range (1..3)')

    if not payload['comment']:
        payload['comment'] = ''

    return payload


# ==================== INFERENCE WITH RETRY ====================

def run_inference(
    model: str, prompt: str, cloud_host: str, key_index: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    """Запуск инференса с обработкой ошибок и замером времени."""
    try:
        t0 = time.perf_counter()
        text = infer_ollama(model, prompt, cloud_host, key_index)
        elapsed_s = time.perf_counter() - t0

        last_err: Optional[Exception] = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                payload = _validate_payload(_json_from_text(text))
                return payload, None, elapsed_s
            except Exception as e:
                last_err = e
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_S * attempt)

        raise last_err or ValueError('Invalid JSON')

    except Exception as e:
        elapsed_s = None
        try:
            if 't0' in locals() and t0 is not None:
                elapsed_s = time.perf_counter() - t0
        except Exception:
            pass
        return None, f'{type(e).__name__}: {e}', elapsed_s


# ==================== PARALLEL MARKUP ====================

@dataclass
class MarkupTask:
    """Задача для разметки одной строки."""
    row_idx: int
    message: str
    topic: str
    llm_context: str
    context_type: str
    key_index: int


def worker_thread(
    queue: Queue,
    df: pd.DataFrame,
    model: str,
    cloud_host: str,
    slot: int,
    context_type: str,
    total: int,
    results: Dict[str, int]
):
    """Поток-воркер для обработки задач из очереди."""
    while True:
        task = queue.get()
        if task is None:
            queue.task_done()
            break

        try:
            r = task.row_idx
            prompt = build_prompt(
                message=task.message,
                topic=task.topic,
                llm_context=task.llm_context,
                context_type=task.context_type
            )

            payload, err, elapsed_s = run_inference(
                model, prompt, cloud_host, task.key_index
            )

            df.at[r, col('LLm', slot)] = model
            if payload is not None:
                df.at[r, col('Экономический эффект', slot)] = payload['economic_effect']
                df.at[r, col('Информационный резонанс', slot)] = payload['information_resonance']
                df.at[r, col('Правильность определения темы', slot)] = payload['topic_agreement']
                df.at[r, col('Экономический нарратив', slot)] = payload['economic_narrative']
                df.at[r, col('Сила нарратива', slot)] = payload['narrative_strength']
                df.at[r, col('Комментарий', slot)] = payload['comment']
                results['ok'] += 1
            else:
                df.at[r, col('Комментарий', slot)] = err or 'Unknown error'
                results['err'] += 1

            tp = f' | {elapsed_s:.2f}s' if elapsed_s is not None else ''
            status = 'OK' if payload is not None else 'ERR'
            err_part = ''
            if payload is None and err:
                err_part = f' | {err[:100]}'

            print(f'Row {r+1}/{total} | key {task.key_index+1} | {status}{tp}{err_part}')

            if CLOUD_THROTTLE_S > 0:
                time.sleep(CLOUD_THROTTLE_S)

        except Exception as e:
            print(f'Row {task.row_idx+1} | FATAL: {e}')
            results['err'] += 1

        finally:
            queue.task_done()


def run_parallel_markup(
    df: pd.DataFrame,
    model: str,
    cloud_host: str,
    slot: int,
    context_type: str,
    start_row: int = 0,
    max_rows: int = 0,
    num_keys: int = 5
) -> pd.DataFrame:
    """
    Параллельная разметка с использованием нескольких API ключей.

    Args:
        df: DataFrame с данными
        model: Имя модели
        cloud_host: Хост Ollama
        slot: Номер слота для разметки
        context_type: Тип контекста ('llm_context', 'llm_context_features', 'llm_context_features_full')
        start_row: Начальная строка
        max_rows: Максимальное количество строк (0 = все)
        num_keys: Количество API ключей (по умолчанию 5)
    """
    total = len(df)
    start = max(0, min(total, start_row))
    end = total if max_rows <= 0 else min(total, start + max_rows)

    # Сбор задач для очереди
    tasks = []
    for r in range(start, end):
        message = _strip(df.at[r, 'message'])
        topic = _strip(df.at[r, 'topic'])

        # Получаем контекст в зависимости от типа
        llm_context = ''
        if context_type in df.columns:
            llm_context = _strip(df.at[r, context_type])

        if not message:
            continue

        # Пропускаем уже размеченные
        if _strip(df.at[r, col('LLm', slot)]):
            print(f'Row {r+1} | уже размечено, пропускаем')
            continue

        # Распределяем по ключам циклически
        key_index = len(tasks) % num_keys
        tasks.append(MarkupTask(
            row_idx=r,
            message=message,
            topic=topic,
            llm_context=llm_context,
            context_type=context_type,
            key_index=key_index
        ))

    if not tasks:
        print('Нет задач для выполнения')
        return df

    print(f'\n=== Запуск параллельной разметки ===')
    print(f'Всего задач: {len(tasks)}')
    print(f'Количество ключей: {num_keys}')
    print(f'Тип контекста: {context_type}')
    print(f'Модель: {model}')
    print(f'Слот: {slot}\n')

    # Создаем очередь и воркеры
    queue = Queue()
    results = {'ok': 0, 'err': 0}

    # Запускаем воркеры (по одному на ключ)
    num_workers = min(num_keys, len(tasks))
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker_thread, args=(queue, df, model, cloud_host, slot, context_type, total, results))
        t.start()
        threads.append(t)

    # Добавляем задачи в очередь
    for task in tasks:
        queue.put(task)

    # Ждем завершения задач
    queue.join()

    # Останавливаем воркеры
    for _ in range(num_workers):
        queue.put(None)
    for t in threads:
        t.join()

    print(f'\n=== Завершено ===')
    print(f'Успешно: {results["ok"]}, Ошибок: {results["err"]}')

    return df


# ==================== ENSURE COLUMNS ====================

def ensure_columns(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Добавление недостающих столбцов для слотов."""
    # Проверка required columns (с поддержкой альтернативных имен)
    if 'message' not in df.columns:
        raise ValueError('Missing required column: message')
    
    # topic может называться topic или topic_llm
    if 'topic' not in df.columns and 'topic_llm' not in df.columns:
        raise ValueError('Missing required column: topic или topic_llm')
    
    # Создаем алиас topic если нужно
    if 'topic_llm' in df.columns and 'topic' not in df.columns:
        df['topic'] = df['topic_llm']

    metric_cols = [
        'LLm',
        'Экономический эффект',
        'Информационный резонанс',
        'Правильность определения темы',
        'Экономический нарратив',
        'Сила нарратива',
        'Комментарий',
    ]

    for i in range(1, n + 1):
        for m in metric_cols:
            cn = col(m, i)
            if cn not in df.columns:
                df[cn] = ''

    for i in range(1, n + 1):
        for m in metric_cols:
            cn = col(m, i)
            df[cn] = df[cn].astype('object').where(~pd.isna(df[cn]), '')

    return df


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Разметка датасета с использованием разных типов контекста'
    )
    parser.add_argument(
        '--context-type',
        type=str,
        default='llm_context',
        choices=['llm_context', 'llm_context_features', 'llm_context_features_full'],
        help='Тип контекста для обогащения (по умолчанию: llm_context)'
    )
    parser.add_argument(
        '--slot',
        type=int,
        default=4,
        help='Номер слота для разметки (по умолчанию: 4)'
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
        default=0,
        help='Максимальное количество строк, 0 = все (по умолчанию: 0)'
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
    args = parser.parse_args()

    print('=' * 60)
    print('Разметка датасета моделью qwen3-vl:235b-instruct-cloud')
    print('с использованием 5 ключей Ollama параллельно')
    print('=' * 60)

    # Проверка API ключей
    valid_keys = len(OLLAMA_API_KEYS)
    if valid_keys == 0:
        print('\nОШИБКА: Не заданы API ключи!')
        print('Установите переменные окружения:')
        print('  OLLAMA_API_KEY_1, OLLAMA_API_KEY_2, OLLAMA_API_KEY_3, OLLAMA_API_KEY_4, OLLAMA_API_KEY_5')
        print('или впишите ключи в скрипт.')
        return

    print(f'\nНайдено валидных ключей: {valid_keys}')
    print(f'Тип контекста: {args.context_type}')

    # Загрузка данных
    input_path = os.path.join(_base_dir, args.input_csv)
    print(f'\nЗагрузка данных из: {input_path}')

    df = pd.read_csv(input_path, encoding='utf-8')
    print(f'  Строк: {len(df)}')
    print(f'  Столбцы: {df.columns.tolist()}')

    # Добавляем столбцы для новой разметки
    df = ensure_columns(df, n=args.slot)

    # Запуск параллельной разметки
    df = run_parallel_markup(
        df=df,
        model=MODEL_NAME,
        cloud_host=OLLAMA_CLOUD_HOST,
        slot=args.slot,
        context_type=args.context_type,
        start_row=args.start_row,
        max_rows=args.max_rows,
        num_keys=min(valid_keys, args.num_keys)
    )

    # Сохранение результатов
    context_suffix = args.context_type.replace('llm_context', '').replace('_', '') or 'base'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = os.path.join(_base_dir, f'goldenset_remarked_{args.context_type}_{stamp}.csv')
    output_xlsx = os.path.join(_base_dir, f'goldenset_remarked_{args.context_type}_{stamp}.xlsx')

    df.to_csv(output_csv, index=False, encoding='utf-8')
    df.to_excel(output_xlsx, index=False)
    print(f'\nРезультаты сохранены:')
    print(f'  CSV: {output_csv}')
    print(f'  XLSX: {output_xlsx}')

    # ==================== ЗАВЕРШЕНИЕ ====================
    print('\n' + '=' * 60)
    print('Завершено!')
    print('=' * 60)
    print('\nДля вычисления метрик и создания графиков запустите:')
    print('  python compute_metrics.py')


if __name__ == '__main__':
    main()
