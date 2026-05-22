# generate_M.py
import asyncio
import json
import os
import torch

from ai.api import run_M_batch_jobs
from ai.prompts import M_prompt


def prepare_tasks(dataset, question_type, model_to_use):
    # --- Task Preparation and Resume after Interruption ---
    all_prompts = M_prompt(dataset_name=dataset, Q=question_type)
    all_tasks = [
        {"id": f"{dataset}_{question_type}_{i}", "prompt_text": p, "pattern": question_type, "llm": model_to_use}
        for i, p in enumerate(all_prompts)
    ]
    return all_tasks


def get_completed_ids(output_file_path):
    completed_ids = set()
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if result.get("status") == "success":
                        completed_ids.add(result.get("id"))
                except json.JSONDecodeError:
                    continue
    return completed_ids


def main():
    dataset = "dblp"
    question_type = "venue"
    model = "gpt-3.5-turbo"

    output_dir = f"gpt_response/metadata/{question_type}"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{dataset}_{question_type}.jsonl")

    all_tasks = prepare_tasks(dataset, question_type, model)
    completed_ids = get_completed_ids(output_file_path)

    tasks_to_run = [task for task in all_tasks if task['id'] not in completed_ids]

    print(f"Total tasks: {len(all_tasks)}. Completed: {len(completed_ids)}. To run: {len(tasks_to_run)}.")

    if tasks_to_run:
        asyncio.run(run_M_batch_jobs(tasks=tasks_to_run, save_path=output_file_path))
    else:
        print("All tasks are already completed.")


if __name__ == '__main__':
    main()