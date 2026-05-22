import asyncio
import json
import os
import torch

from ai.api import run_R_E_batch_jobs
from ai.prompts import R_E_prompt


def get_all_raw(dataset_path):
    data = torch.load(dataset_path)
    dataset = dataset_path.split("/")[-1].split("_")[0]

    if dataset == "dblp":
        return [
            {"idx": idx, "raw_text": text}
            for idx, text in enumerate(data.raw_texts)
        ]

    if dataset == "arxiv":
        return [
            {"idx": idx, "raw_text": f"Abstract: {data.abs[idx]}\nTitle: {data.title[idx]}"}
            for idx in range(len(data.abs))
        ]

    return None



def generate_entity(dataset_path, save_path, focus='R', llm='gpt-3.5-turbo'):

    dataset = dataset_path.split("/")[-1].split("_")[0]

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

    raw_texts = get_all_raw(dataset_path)
    all_prompts = R_E_prompt(dataset, raw_texts, focus)
    tasks_to_run = [p for p in all_prompts if p['id'] not in completed_ids]

    if not tasks_to_run:
        print("All tasks are already completed. Nothing to do.")
        return

    print(f"Total tasks: {len(all_prompts)}, Tasks to run: {len(tasks_to_run)}")

    tasks = [{'prompt': prompt, 'pattern': focus, 'llm': llm} for prompt in tasks_to_run]
    asyncio.run(run_R_E_batch_jobs(tasks, save_path))


if __name__ == '__main__':
    DATASET_FILE = 'preprocessed_data/new/dblp_random_sbert.pt'
    Pattern = 'R'
    LLM_MODEL = 'gpt'

    SAVE_DIR = 'gpt_response/entity'
    os.makedirs(SAVE_DIR, exist_ok=True)

    d_name = DATASET_FILE.split('/')[-1].split('_')[0]
    OUTPUT_FILE = os.path.join(SAVE_DIR, f'{d_name}_{Pattern}_{LLM_MODEL}.jsonl')


    generate_entity(
        dataset_path=DATASET_FILE,
        save_path=OUTPUT_FILE,
        focus=Pattern,
        llm=LLM_MODEL
    )
