import asyncio
import csv
import json
import os

from ai.api import ShuiHu_batch_jobs
from ai.prompts import ShuiHu_prompt


def get_all_raw_ShuiHu(dataset_path, text_select):
    items = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(text_select) and row.get('PERSON_ID'):
                items.append({
                    "id": row['PERSON_ID'],
                    "text": row[text_select]
                })
    return items


def generate_ShuiHu(dataset_path, save_path, text_select, llm='gpt-3.5-turbo'):
    completed_ids = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('status') == 'success':
                        completed_ids.add(data['id'])
                except (json.JSONDecodeError, KeyError):
                    continue

    if completed_ids:
        print(f"Found {len(completed_ids)} completed tasks. Resuming...")

    raw_items = get_all_raw_ShuiHu(dataset_path, text_select)
    all_prompts = ShuiHu_prompt(raw_items)
    tasks_to_run = [p for p in all_prompts if p['id'] not in completed_ids]

    if not tasks_to_run:
        print("All tasks are already completed. Nothing to do.")
        return

    print(f"Total tasks: {len(all_prompts)}, Tasks to run: {len(tasks_to_run)}")

    tasks = [{'prompt': prompt, 'llm': llm} for prompt in tasks_to_run]
    asyncio.run(ShuiHu_batch_jobs(tasks, save_path))

if __name__ == '__main__':
    DATASET_FILE = 'dataset/ShuiHu/nodes.csv'
    LLM_MODEL = 'gpt'
    TEXT_SELECT = 'DES'

    SAVE_DIR = 'gpt_response/ShuiHu'
    os.makedirs(SAVE_DIR, exist_ok=True)

    d_name = DATASET_FILE.split('/')[-1].split('.')[0]
    OUTPUT_FILE = os.path.join(SAVE_DIR, f'{d_name}_{TEXT_SELECT}_{LLM_MODEL}.jsonl')

    generate_ShuiHu(
        dataset_path=DATASET_FILE,
        save_path=OUTPUT_FILE,
        text_select=TEXT_SELECT,
        llm=LLM_MODEL
    )