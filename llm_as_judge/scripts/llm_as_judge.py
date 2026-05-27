"""
LLM as Judge - Система оценки классификаторов экономических нарративов
Судьи: minimax-m2.5:cloud, deepseek-v3.2:cloud, qwen3.5:cloud glm-5:cloud kimi-k2.5:cloud
"""

import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
from ollama import chat
from datetime import datetime
import re
import time
import pickle


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# API keys are loaded only from environment variables.
# Never commit provider keys or local config files into this repository.
OLLAMA_API_KEYS = [
    key for key in (
        os.environ.get('OLLAMA_API_KEY_1'),
        os.environ.get('OLLAMA_API_KEY_2'),
        os.environ.get('OLLAMA_API_KEY_3'),
        os.environ.get('OLLAMA_API_KEY_4'),
    )
    if key
]

JUDGE_MODELS = [
    # 'minimax-m2.5:cloud',  
    # 'deepseek-v3.2:cloud',
    'qwen3.5:397b-cloud',
    'glm-5:cloud',
    'kimi-k2.5:cloud'

]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKUP_FILE = os.environ.get('MARKUP_FILE', os.path.join(BASE_DIR, 'LLm markup.xlsx'))
GOLDEN_FILE = os.environ.get('GOLDEN_FILE', os.path.join(BASE_DIR, 'golden_set_v1.csv'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', BASE_DIR)

MAX_FAILURES = 3
RATE_WAIT = 90
TIMEOUT = 60

# Текущий индекс API ключа (для ротации)
_api_key_index = 0


def get_api_key():
    """Получение текущего API ключа с ротацией"""
    global _api_key_index
    if not OLLAMA_API_KEYS:
        return None
    key = OLLAMA_API_KEYS[_api_key_index % len(OLLAMA_API_KEYS)]
    _api_key_index += 1
    return key


# ============================================================================
# ПРОМПТ СУДЬИ
# ============================================================================

JUDGE_PROMPT = """Ты — судья для сравнения классификаторов экономических нарративов.

Сравни 6 LLM с эталоном человека. Определи лучшего для данной новости.

ВАЖНО: Наибольший вес — точность определения нарратива (Да/Нет). Если нарратив неверен, остальные критерии вторичны.

Критерии (0-10):
1. narrative: 10=точное Да/Нет (НАИБОЛЕЕ ВАЖНЫЙ КРИТЕРИЙ)
2. strength: 10=точное число
3. effect: 10=точное или ±1
4. resonance: 10=точное
5. topic: 10=тема верна

Верни ТОЛЬКО JSON без markdown:
{{
  "scores": {{
    "human": {{"total": 50, "narrative": 10, "strength": 10, "effect": 10, "resonance": 10, "topic": 10}},
    "llm_1": {{"total": 45, "narrative": 9, "strength": 8, "effect": 10, "resonance": 9, "topic": 9}},
    "llm_2": {{"total": 40, "narrative": 8, "strength": 7, "effect": 9, "resonance": 8, "topic": 8}},
    "llm_3": {{"total": 35, "narrative": 7, "strength": 6, "effect": 8, "resonance": 7, "topic": 7}},
    "llm_4": {{"total": 30, "narrative": 6, "strength": 5, "effect": 7, "resonance": 6, "topic": 6}},
    "llm_5": {{"total": 25, "narrative": 5, "strength": 4, "effect": 6, "resonance": 5, "topic": 5}},
    "llm_6": {{"total": 20, "narrative": 4, "strength": 3, "effect": 5, "resonance": 4, "topic": 3}}
  }},
  "ranking": ["llm_1", "human", "llm_2", "llm_3", "llm_4", "llm_5", "llm_6"],
  "best": {{"winner": "llm_1", "score": 45, "reason": "Наиболее точные оценки"}}
}}

Данные:
Новость: {message}
Тема: {topic}

Человек: narrative={h_narr}, strength={h_str}, effect={h_eff}, resonance={h_res}, topic={h_top}

LLM 1: narrative={l1_narr}, strength={l1_str}, effect={l1_eff}, resonance={l1_res}, topic={l1_top}
LLM 2: narrative={l2_narr}, strength={l2_str}, effect={l2_eff}, resonance={l2_res}, topic={l2_top}
LLM 3: narrative={l3_narr}, strength={l3_str}, effect={l3_eff}, resonance={l3_res}, topic={l3_top}
LLM 4: narrative={l4_narr}, strength={l4_str}, effect={l4_eff}, resonance={l4_res}, topic={l4_top}
LLM 5: narrative={l5_narr}, strength={l5_str}, effect={l5_eff}, resonance={l5_res}, topic={l5_top}
LLM 6: narrative={l6_narr}, strength={l6_str}, effect={l6_eff}, resonance={l6_res}, topic={l6_top}

JSON:"""


# ============================================================================
# ELO СИСТЕМА
# ============================================================================

class ELOTracker:
    """Отслеживание ELO рейтинга классификаторов"""
    
    def __init__(self):
        self.ratings = {
            'human': 1500,
            'llm_1': 1500, 'llm_2': 1500, 'llm_3': 1500,
            'llm_4': 1500, 'llm_5': 1500, 'llm_6': 1500
        }
        self.wins = {k: 0 for k in self.ratings}
        self.matches = {k: 0 for k in self.ratings}
        self.changes = {k: 0 for k in self.ratings}  # Изменения за последнюю новость
    
    def update(self, ranking: List[str], scores: Dict):
        """Обновление ELO после одной новости"""
        K = 32
        prev_ratings = self.ratings.copy()
        
        for i, player in enumerate(ranking):
            if player not in self.ratings:
                continue
            self.matches[player] += 1
            
            for j, other in enumerate(ranking):
                if i == j or other not in self.ratings:
                    continue
                
                expected = 1 / (1 + 10 ** ((self.ratings[other] - self.ratings[player]) / 400))
                actual = 1 if scores.get(player, {}).get('total', 0) > scores.get(other, {}).get('total', 0) else 0
                if scores.get(player, {}).get('total', 0) == scores.get(other, {}).get('total', 0):
                    actual = 0.5
                
                delta = K * (actual - expected)
                self.ratings[player] += delta
            
            if i == 0:
                self.wins[player] += 1
        
        # Сохраняем изменения
        for k in self.ratings:
            self.changes[k] = round(self.ratings[k] - prev_ratings[k], 1)
    
    def get_standings(self) -> List[Dict]:
        """Таблица лидеров"""
        standings = []
        for key in self.ratings:
            standings.append({
                'name': key,
                'rating': round(self.ratings[key], 1),
                'wins': self.wins[key],
                'matches': self.matches[key],
                'win_rate': round(self.wins[key] / max(1, self.matches[key]) * 100, 1)
            })
        return sorted(standings, key=lambda x: x['rating'], reverse=True)
    
    def get_changes(self) -> Dict[str, float]:
        """Возвращает изменения ELO за последнюю новость"""
        return self.changes.copy()


# ============================================================================
# ФУНКЦИИ
# ============================================================================

def normalize(val) -> str:
    """Нормализация значений"""
    if pd.isna(val):
        return 'N/A'
    if isinstance(val, (int, float)):
        return str(int(float(val))) if float(val) == int(float(val)) else str(val)
    return str(val)


def call_judge(message: str, topic: str, human: Dict, llms: Dict[str, Dict],
               judge_model: str, retry: int = 0) -> Tuple[Optional[Dict], bool]:
    """Вызов судьи"""
    from ollama import Client

    prompt = JUDGE_PROMPT.format(
        message=message[:600],
        topic=topic,
        h_narr=human.get('narrative', 'N/A'),
        h_str=human.get('strength', 'N/A'),
        h_eff=human.get('effect', 'N/A'),
        h_res=human.get('resonance', 'N/A'),
        h_top=human.get('topic', 'N/A'),
        l1_narr=llms.get('llm_1', {}).get('narrative', 'N/A'),
        l1_str=llms.get('llm_1', {}).get('strength', 'N/A'),
        l1_eff=llms.get('llm_1', {}).get('effect', 'N/A'),
        l1_res=llms.get('llm_1', {}).get('resonance', 'N/A'),
        l1_top=llms.get('llm_1', {}).get('topic', 'N/A'),
        l2_narr=llms.get('llm_2', {}).get('narrative', 'N/A'),
        l2_str=llms.get('llm_2', {}).get('strength', 'N/A'),
        l2_eff=llms.get('llm_2', {}).get('effect', 'N/A'),
        l2_res=llms.get('llm_2', {}).get('resonance', 'N/A'),
        l2_top=llms.get('llm_2', {}).get('topic', 'N/A'),
        l3_narr=llms.get('llm_3', {}).get('narrative', 'N/A'),
        l3_str=llms.get('llm_3', {}).get('strength', 'N/A'),
        l3_eff=llms.get('llm_3', {}).get('effect', 'N/A'),
        l3_res=llms.get('llm_3', {}).get('resonance', 'N/A'),
        l3_top=llms.get('llm_3', {}).get('topic', 'N/A'),
        l4_narr=llms.get('llm_4', {}).get('narrative', 'N/A'),
        l4_str=llms.get('llm_4', {}).get('strength', 'N/A'),
        l4_eff=llms.get('llm_4', {}).get('effect', 'N/A'),
        l4_res=llms.get('llm_4', {}).get('resonance', 'N/A'),
        l4_top=llms.get('llm_4', {}).get('topic', 'N/A'),
        l5_narr=llms.get('llm_5', {}).get('narrative', 'N/A'),
        l5_str=llms.get('llm_5', {}).get('strength', 'N/A'),
        l5_eff=llms.get('llm_5', {}).get('effect', 'N/A'),
        l5_res=llms.get('llm_5', {}).get('resonance', 'N/A'),
        l5_top=llms.get('llm_5', {}).get('topic', 'N/A'),
        l6_narr=llms.get('llm_6', {}).get('narrative', 'N/A'),
        l6_str=llms.get('llm_6', {}).get('strength', 'N/A'),
        l6_eff=llms.get('llm_6', {}).get('effect', 'N/A'),
        l6_res=llms.get('llm_6', {}).get('resonance', 'N/A'),
        l6_top=llms.get('llm_6', {}).get('topic', 'N/A'),
    )

    try:
        api_key = get_api_key()
        if api_key:
            print(f"    Вызов {judge_model} (ключ #{OLLAMA_API_KEYS.index(api_key) + 1})...")
            client = Client(host='https://ollama.com', headers={'Authorization': f'Bearer {api_key}'})
            response = client.chat(
                model=judge_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'timeout': TIMEOUT}
            )
            content = response['message']['content']
        else:
            print(f"    Вызов {judge_model} (без ключа)...")
            response = chat(
                model=judge_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'timeout': TIMEOUT}
            )
            content = response.message.content
        
        print(f"    Получено ({len(content)} символов)")

        if not content or len(content.strip()) == 0:
            print("    Пустой ответ")
            return None, True

        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            return result, False
        else:
            print(f"Нет JSON: {content[:200]}")
            return None, True

    except json.JSONDecodeError as e:
        print(f"JSON ошибка: {e}")
        return None, True
    except Exception as e:
        err = str(e)
        print(f"Ошибка: {err}")
        if any(x in err.lower() for x in ['timeout', 'limit', 'rate', 'connection']):
            return None, True
        return None, False


def save_checkpoint(path: str, results: List, row: int, elo: ELOTracker):
    try:
        with open(path, 'wb') as f:
            pickle.dump({
                'results': results,
                'row': row,
                'elo_ratings': elo.ratings,
                'elo_wins': elo.wins,
                'elo_matches': elo.matches
            }, f)
        print("    Чекпоинт сохранён")
    except Exception as e:
        print(f"Ошибка: {e}")


def load_checkpoint(path: str):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                print(f"Загружен чекпоинт")
                elo = ELOTracker()
                elo.ratings = data.get('elo_ratings', elo.ratings)
                elo.wins = data.get('elo_wins', elo.wins)
                elo.matches = data.get('elo_matches', elo.matches)
                return data.get('results', []), data.get('row', 0), elo
        except Exception as e:
            print(f"Ошибка: {e}")
    return [], 0, ELOTracker()


def save_results(results: List, path: str, markup_df: pd.DataFrame, judge_model: str, elo: ELOTracker = None):
    """Сохранение результатов в Excel"""
    flat = []

    for i, r in enumerate(results):
        judge = r.get('judge_result', {})
        if not judge:
            continue

        scores = judge.get('scores', {})
        ranking = judge.get('ranking', [])
        best = judge.get('best', {})

        # Берём сообщение напрямую из markup_df
        message = markup_df.iloc[i]['message'] if i < len(markup_df) else r.get('message', '')

        flat.append({
            'message': str(message)[:500],
            'topic': r.get('topic', ''),
            'human_total': scores.get('human', {}).get('total', 0),
            'llm_1_total': scores.get('llm_1', {}).get('total', 0),
            'llm_2_total': scores.get('llm_2', {}).get('total', 0),
            'llm_3_total': scores.get('llm_3', {}).get('total', 0),
            'llm_4_total': scores.get('llm_4', {}).get('total', 0),
            'llm_5_total': scores.get('llm_5', {}).get('total', 0),
            'llm_6_total': scores.get('llm_6', {}).get('total', 0),
            'ranking': ' > '.join(ranking),
            'best': best.get('winner', ''),
            'best_score': best.get('score', 0),
            'best_reason': best.get('reason', ''),
        })

    df = pd.DataFrame(flat)

    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Результаты')

        # Сводка по средним баллам
        if len(df) > 0:
            summary_data = []
            classifiers = ['human'] + [f'llm_{i}' for i in range(1, 7)]
            names = {'human': 'Человек'} | {f'llm_{i}': f'LLM {i}' for i in range(1, 7)}

            for cl in classifiers:
                total_col = f'{cl}_total'
                # Считаем средние по критериям из scores
                avg_narr = sum(r.get('judge_result', {}).get('scores', {}).get(cl, {}).get('narrative', 0) for r in results) / max(1, len(results))
                avg_str = sum(r.get('judge_result', {}).get('scores', {}).get(cl, {}).get('strength', 0) for r in results) / max(1, len(results))
                avg_eff = sum(r.get('judge_result', {}).get('scores', {}).get(cl, {}).get('effect', 0) for r in results) / max(1, len(results))
                avg_res = sum(r.get('judge_result', {}).get('scores', {}).get(cl, {}).get('resonance', 0) for r in results) / max(1, len(results))
                avg_top = sum(r.get('judge_result', {}).get('scores', {}).get(cl, {}).get('topic', 0) for r in results) / max(1, len(results))
                
                summary_data.append({
                    'Классификатор': names.get(cl, cl),
                    'Средний балл': round(df[total_col].mean(), 2),
                    'Побед': (df['best'] == cl).sum(),
                    'Макс': df[total_col].max(),
                    'Мин': df[total_col].min(),
                    'Средний narrative': round(avg_narr, 2),
                    'Средний strength': round(avg_str, 2),
                    'Средний effect': round(avg_eff, 2),
                    'Средний resonance': round(avg_res, 2),
                    'Средний topic': round(avg_top, 2),
                })
            summary = pd.DataFrame(summary_data)
            summary.to_excel(w, index=False, sheet_name='Сводка')

            # ELO таблица
            if elo:
                elo_standings = elo.get_standings()
                elo_data = []
                for s in elo_standings:
                    elo_data.append({
                        'Классификатор': names.get(s['name'], s['name']),
                        'Итоговое ELO': s['rating'],
                        'Побед': s['wins'],
                        'Матчей': s['matches'],
                        'Win Rate %': s['win_rate'],
                    })
                elo_df = pd.DataFrame(elo_data)
                elo_df.to_excel(w, index=False, sheet_name='ELO Рейтинг')
                
                # Изменения ELO по ходу оценки
                elo_history = []
                for idx, r in enumerate(results):
                    changes = r.get('judge_result', {}).get('elo_changes', {})
                    if changes:
                        elo_history.append({
                            'Строка': idx + 1,
                            **{names.get(k, k): v for k, v in changes.items()}
                        })
                if elo_history:
                    elo_hist_df = pd.DataFrame(elo_history)
                    elo_hist_df.to_excel(w, index=False, sheet_name='ELO Изменения')

    print(f"Сохранено {len(flat)} результатов в {path}")


def run_evaluation(markup_path: str, golden_path: str, output_path: str,
                   judge_model: str, max_rows: int = 500):
    """Оценка с использованием конкретной модели судьи"""
    
    print(f"\n{'='*60}")
    print(f"СУДЬЯ: {judge_model}")
    print(f"{'='*60}")
    
    print("Загрузка данных...")
    markup_df = pd.read_excel(markup_path)
    golden_df = pd.read_csv(golden_path)
    
    print(f"Разметка: {len(markup_df)} строк")
    print(f"Эталон: {len(golden_df)} строк")
    
    results = []
    start_row = 0
    elo = ELOTracker()
    
    # Чекпоинт для каждой модели свой
    checkpoint_path = output_path.replace('.xlsx', '_checkpoint.pkl')
    
    if os.path.exists(checkpoint_path):
        results, start_row, elo = load_checkpoint(checkpoint_path)
        if results:
            print(f"Возобновление с строки {start_row + 1}")
    
    eval_rows = min(max_rows, len(markup_df), len(golden_df))
    failures = 0
    
    for idx in range(start_row, eval_rows):
        row = markup_df.iloc[idx]
        golden = golden_df.iloc[idx]
        
        print(f"\n{'='*50}")
        print(f"Строка {idx + 1}/{eval_rows}")
        print(f"Тема: {row.get('topic', 'N/A')}")
        msg_preview = str(row.get('message', ''))[:60].replace('\n', ' ')
        print(f"Сообщение: {msg_preview}...")
        
        # Данные человека
        human = {
            'narrative': 'Да' if golden.get('economic_narrative', 0) == 1 else 'Нет',
            'strength': normalize(golden.get('narrative_strength', 0)),
            'effect': normalize(golden.get('economic_effect', 0)),
            'resonance': normalize(golden.get('information_resonance', 0)),
            'topic': '3',
        }
        
        # Данные LLM
        llms = {}
        for i in range(1, 7):
            llms[f'llm_{i}'] = {
                'narrative': normalize(row.get(f'Экономический нарратив ({i})', 'N/A')),
                'strength': normalize(row.get(f'Сила нарратива ({i})', 'N/A')),
                'effect': normalize(row.get(f'Экономический эффект ({i})', 'N/A')),
                'resonance': normalize(row.get(f'Информационный резонанс ({i})', 'N/A')),
                'topic': normalize(row.get(f'Правильность определения темы ({i})', 'N/A')),
            }
        
        # Вызов судьи
        result = None
        retry = 0
        while retry < 3:
            res, should_retry = call_judge(
                message=row.get('message', ''),
                topic=row.get('topic', ''),
                human=human,
                llms=llms,
                judge_model=judge_model,
                retry=retry
            )
            if res:
                result = res
                break
            if not should_retry:
                break
            retry += 1
            if retry < 3:
                wait = 5 * retry
                print(f"    Повтор через {wait}с...")
                time.sleep(wait)
        
        if result:
            ranking = result.get('ranking', [])
            best = result.get('best', {}).get('winner', 'N/A')
            elo.update(ranking, result.get('scores', {}))
            result['elo_changes'] = elo.get_changes()
            print(f"    Лучший: {best}")
            print(f"    Ранкинг: {' > '.join(ranking)}")
            failures = 0
        else:
            print("    Оценка не удалась")
            failures += 1
            result = {'elo_changes': {}}
        
        results.append({
            'message': row.get('message', ''),
            'topic': row.get('topic', ''),
            'judge_result': result
        })
        
        # Чекпоинт
        if checkpoint_path:
            save_checkpoint(checkpoint_path, results, idx, elo)
        
        # Бэкап каждые 10 строк
        if (idx + 1) % 10 == 0:
            backup = output_path.replace('.xlsx', f'_backup_row{idx + 1}.xlsx')
            try:
                save_results(results, backup, markup_df, judge_model, elo)
                print(f"    Бэкап на строке {idx + 1}")
            except Exception as e:
                print(f"    Ошибка бэкапа: {e}")
        
        # Лимит
        if failures >= MAX_FAILURES:
            print(f"\n{'!'*60}")
            print(f"ЛИМИТ: {failures} ошибок подряд")
            print(f"Ожидание {RATE_WAIT} мин...")
            print(f"{'!'*60}\n")
            time.sleep(RATE_WAIT * 60)
            failures = 0
    
    # Финальное сохранение
    save_results(results, output_path, markup_df, judge_model, elo)
    
    # Финальная статистика ELO
    print(f"\n{'='*60}")
    print(f"ФИНАЛЬНАЯ СТАТИСТИКА ELO ({judge_model})")
    print(f"{'='*60}")
    standings = elo.get_standings()
    for i, s in enumerate(standings, 1):
        print(f"{i}. {s['name']:10} ELO: {s['rating']:7.1f} Побед: {s['wins']:3} Из {s['matches']} ({s['win_rate']}%)")
    
    winner = standings[0]
    print(f"\nЛУЧШИЙ: {winner['name']} (ELO: {winner['rating']}, Побед: {winner['wins']}/{winner['matches']})")
    
    # Удаление чекпоинта
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Чекпоинт удалён")
    
    return results, elo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("LLM as Judge - Сравнение классификаторов")
    print("="*60)
    print(f"Судьи: {JUDGE_MODELS}")
    print(f"Строк: 500")
    
    all_results = {}
    
    for judge_model in JUDGE_MODELS:
        output_file = os.path.join(OUTPUT_DIR, f'judge_results_{judge_model.replace(":", "_").replace(".", "_")}.xlsx')
        results, elo = run_evaluation(MARKUP_FILE, GOLDEN_FILE, output_file, judge_model, max_rows=500)
        all_results[judge_model] = (results, elo)
    
    print("\n" + "="*60)
    print("ВСЕ ОЦЕНКИ ЗАВЕРШЕНЫ!")
    print("="*60)
    
    # Итоговая сводка по всем судьям
    for judge_model, (results, elo) in all_results.items():
        print(f"\n{judge_model}:")
        standings = elo.get_standings()
        winner = standings[0]
        print(f"  Победитель: {winner['name']} (ELO: {winner['rating']})")
