import time

import httpx
import openai
from openai import OpenAI, AsyncOpenAI

import asyncio
import json
from typing import List, Dict
from tqdm import tqdm as sync_tqdm
import aiohttp

from core.utils import load_secret


CONCURRENCY_LIMIT = 3
SEMAPHORE = asyncio.Semaphore(CONCURRENCY_LIMIT)
file_lock = asyncio.Lock()

async def async_openai_text_api(messages,
                    api_key ='',
                    model="gpt-3.5-turbo",
                    temperature=0, max_tokens=100):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 100.

    Returns:
        str: Response (dict).
    """

    response =await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    return response


async def async_process_three_prompts(input_text, api_key, author_q, venue_q, abstract_q):

    if author_q:
        max_tokens = 100
    elif abstract_q:
        max_tokens = 400
    elif venue_q:
        max_tokens = 50
    else:
        max_tokens = 100

    messages = [{"role": "user", "content": input_text}]

    response = await async_openai_text_api(messages, api_key, max_tokens=max_tokens)

    return response

def openrouter_gpt_api_0613(messages,
                        api_key,
                        model="openai/gpt-3.5-turbo-0613",
                        temperature=0, max_tokens=100,
                        ):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 100.

    Returns:
        str: Response (dict).
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response

def openrouter_gpt_api(messages,
                        api_key,
                        model="openai/gpt-3.5-turbo",
                        temperature=0, max_tokens=100,
                        ):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "gpt-3.5-turbo".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 100.

    Returns:
        str: Response (dict).
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response

def qwen_api(messages,
            api_key,
            model="qwen3-30b-a3b-instruct-2507",
            temperature=0, max_tokens=100,
                        ):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "qwen/qwen3-30b-a3b:free".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 100.

    Returns:
        str: Response (dict).
    """

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response

def qwen_plus_api(messages,
            api_key,
            model="qwen-plus",
            temperature=0, max_tokens=100,
                        ):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "qwen/qwen3-30b-a3b:free".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 100.

    Returns:
        str: Response (dict).
    """

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response




def official_deepseek_api(messages,
                api_key,
                model="deepseek-chat",
                temperature=0, max_tokens=1000,
                ):
    """
    Get completion from the OpenAI API based on the given messages.

    Parameters:
        messages (list): Messages to be sent to the OpenAI API.
        model (str, optional): The name of the model to be used. Default is "deepseek-chat".
        temperature (float, optional): Sampling temperature. Default is 0.
        max_tokens (int, optional): Maximum number of tokens for the response. Default is 100.

    Returns:
        str: Response (dict).
    """

    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response

# DL
def process_three_prompts(model, input_text, author_q, abstract_q, venue_q):

    if author_q:
        max_tokens = 500
    elif abstract_q:
        max_tokens = 1000
    elif venue_q:
        max_tokens = 500
    else:
        max_tokens = 500

    if model == "gpt-3.5-turbo":

        api_key = load_secret()['openai']['secret']
        messages = [{"role": "user", "content": input_text}]
        print(f"[PROCESS] Processing input: {input_text[:40]}...")
        response = openrouter_gpt_api_0613(messages, api_key, max_tokens=max_tokens)
        response_dict = response.to_dict()

    elif model == "qwen-plus":

        api_key = load_secret()['tongyi']['secret']
        messages = [{"role": "user", "content": input_text}]
        print(f"[PROCESS] Processing input: {input_text[:40]}...")
        response = qwen_plus_api(messages, api_key, max_tokens=max_tokens)
        response_dict = response.to_dict()

    elif model == "deepseek-chat":

        api_key = load_secret()['deepseek']['secret']
        messages = [{"role": "user", "content": input_text}]
        print(f"[PROCESS] Processing input: {input_text[:40]}...")
        response = official_deepseek_api(messages, api_key, max_tokens=max_tokens)
        response_dict = response.to_dict()

    return response_dict


async def worker(session: aiohttp.ClientSession, task: Dict, save_path: str):
    """
    A work unit for a single API request.
    It includes rate limiting, invoking the API, and writing the result (success or failure) to a file.
    """
    async with SEMAPHORE:
        try:

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                process_three_prompts,
                task['model_name'],
                task['prompt']['prompt'],
                task['author_q'],
                task['abstract_q'],
                task['venue_q']
            )

            response["pid"] = task['prompt']["pid"]
            response["title"] = task['prompt']["title"]
            if "label" in task['prompt']:
                response["label"] = task['prompt']["label"]
            response["prompt"] = task['prompt']["prompt"]
            response["status"] = "success"  # 添加成功状态

            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")

            return {"pid": response["pid"], "status": "success"}

        except Exception as e:
            error_message = {
                "pid": task['prompt']["pid"],
                "status": "failed",
                "error": str(e),
                "prompt": task['prompt']
            }
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_message, ensure_ascii=False) + "\n")
            return error_message


async def run_batch_jobs(tasks: List[Dict], save_path: str):
    """
    The main entry point for asynchronous tasks.
    The tasks are processed in chunks, and after completing each batch of size CONCURRENCY_LIMIT,
    the process pauses for 10 seconds before moving on to the next batch.
    """
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    async with aiohttp.ClientSession() as session:
        with sync_tqdm(total=len(tasks), desc="Processing API requests") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch_tasks_data = tasks[i:i + batch_size]

                async_tasks_batch = [worker(session, task, save_path) for task in batch_tasks_data]

                batch_results = await asyncio.gather(*async_tasks_batch)
                all_results.extend(batch_results)

                pbar.update(len(batch_tasks_data))

                if i + batch_size < len(tasks):
                    pbar.set_description(f"Batch complete. Sleeping for {sleep_interval}s...")
                    await asyncio.sleep(sleep_interval)
                    pbar.set_description("Processing API requests")

        success_count = sum(1 for r in all_results if r and r.get("status") == "success")
        failed_count = len(all_results) - success_count
        print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")


def process_R_E(input_text, pattern, llm = 'gpt'):

    max_tokens = 2000

    if llm in "gpt-3.5-turbo":
        api_key = load_secret()['openai']['secret']
        call_api = openrouter_gpt_api_0613
    elif llm in "qwen-plus":
        api_key = load_secret()['tongyi']['secret']
        call_api = qwen_plus_api
    elif llm in "deepseek-chat":
        api_key = load_secret()['deepseek']['secret']
        call_api = official_deepseek_api
    else:
        raise ValueError(f"Unsupported llm: {llm}")

    messages = [{"role": "user", "content": input_text}]
    print(f"[PROCESS-{pattern}] Processing input: {input_text[:40]}...")

    response = call_api(messages, api_key, max_tokens=max_tokens)
    response_dict = response.to_dict()

    return response_dict

async def R_E_worker(session: aiohttp.ClientSession, task: Dict, save_path: str):
    async with SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, process_R_E, task['prompt']['prompt'], task['pattern'], task['llm']
            )

            response["id"] = task['prompt']["id"]
            response["prompt"] = task['prompt']["prompt"]
            response["status"] = "success"

            return response

        except Exception as e:
            error_message = {
                "id": task['prompt']["id"],
                "status": "failed",
                "error": str(e),
                "prompt": task['prompt']
            }
            return error_message


async def run_R_E_batch_jobs(tasks: List[Dict], save_path: str):
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    async with aiohttp.ClientSession() as session:
        with sync_tqdm(total=len(tasks), desc="Processing R/E API requests") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch_tasks_data = tasks[i:i + batch_size]
                async_tasks_batch = [R_E_worker(session, task, save_path) for task in batch_tasks_data]
                batch_results = await asyncio.gather(*async_tasks_batch)

                async with file_lock:
                    with open(save_path, "a", encoding="utf-8") as f:
                        for res in batch_results:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

                all_results.extend(batch_results)

                pbar.update(len(batch_tasks_data))

                if i + batch_size < len(tasks):
                    pbar.set_description(f"Batch complete. Sleeping for {sleep_interval}s...")
                    await asyncio.sleep(sleep_interval)
                    pbar.set_description("Processing R/E API requests")

        success_count = sum(1 for r in all_results if r and r.get("status") == "success")
        failed_count = len(all_results) - success_count
        print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")

# M
def process_M(input_text, Q, llm):
    if Q in ["venue", "v"]:
        max_tokens = 1000
    elif Q in ["author", "a"]:
        max_tokens = 500
    else:
        max_tokens = 800

    if llm == "gpt-3.5-turbo":
        api_key = load_secret()['openai']['secret']
        call_api = openrouter_gpt_api_0613
    elif llm == "qwen-plus":
        api_key = load_secret()['tongyi']['secret']
        call_api = qwen_plus_api
    elif llm == "deepseek-chat":
        api_key = load_secret()['deepseek']['secret']
        call_api = official_deepseek_api
    else:
        raise ValueError(f"Unsupported llm: {llm}")

    messages = [{"role": "user", "content": input_text}]
    print(f"[PROCESS-{Q}] Processing input: {input_text[:40]}...")

    response = call_api(messages, api_key, max_tokens=max_tokens)
    response_dict = response.to_dict()

    return response_dict


async def M_worker(task: Dict):
    async with SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, process_M, task['prompt_text'], task['pattern'], task['llm']
            )

            response["id"] = task['id']
            response["prompt"] = task['prompt_text']
            response["status"] = "success"

            return response

        except Exception as e:
            error_message = {
                "id": task['id'],
                "status": "failed",
                "error": str(e),
                "prompt": task['prompt_text']
            }
            return error_message


async def run_M_batch_jobs(tasks: List[Dict], save_path: str):
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    with sync_tqdm(total=len(tasks), desc="Processing M API requests") as pbar:
        for i in range(0, len(tasks), batch_size):
            batch_tasks_data = tasks[i:i + batch_size]

            async_tasks_batch = [M_worker(task) for task in batch_tasks_data]

            batch_results = await asyncio.gather(*async_tasks_batch)

            async with file_lock:
                with open(save_path, "a", encoding="utf-8") as f:
                    for res in batch_results:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")

            all_results.extend(batch_results)
            pbar.update(len(batch_tasks_data))

            if i + batch_size < len(tasks):
                pbar.set_description(f"Batch complete. Sleeping for {sleep_interval}s...")
                await asyncio.sleep(sleep_interval)
                pbar.set_description("Processing M API requests")

    success_count = sum(1 for r in all_results if r and r.get("status") == "success")
    failed_count = len(all_results) - success_count
    print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")


def process_ShuiHu(input_text, llm='gpt'):
    max_tokens = 1000

    if llm == "gpt-3.5-turbo" or llm == "gpt":
        api_key = load_secret()['openai']['secret']
        call_api = openrouter_gpt_api_0613
    elif llm == "qwen-plus":
        api_key = load_secret()['tongyi']['secret']
        call_api = qwen_plus_api
    elif llm == "deepseek-chat":
        api_key = load_secret()['deepseek']['secret']
        call_api = official_deepseek_api
    else:
        raise ValueError(f"Unsupported llm: {llm}")

    messages = [{"role": "user", "content": input_text}]
    print(f"[PROCESS-ShuiHu] Processing input: {input_text[:40]}...")

    response = call_api(messages, api_key, max_tokens=max_tokens)
    response_dict = response.to_dict()

    return response_dict

async def ShuiHu_worker(session: aiohttp.ClientSession, task: Dict, save_path: str):
    async with SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, process_ShuiHu, task['prompt']['prompt'], task['llm']
            )

            response["id"] = task['prompt']["id"]
            response["prompt"] = task['prompt']["prompt"]
            response["status"] = "success"

            return response

        except Exception as e:
            error_message = {
                "id": task['prompt']["id"],
                "status": "failed",
                "error": str(e),
                "prompt": task['prompt']
            }
            return error_message

async def ShuiHu_batch_jobs(tasks: List[Dict], save_path: str):
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    async with aiohttp.ClientSession() as session:
        with sync_tqdm(total=len(tasks), desc="Processing ShuiHu API requests") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch_tasks_data = tasks[i:i + batch_size]
                async_tasks_batch = [ShuiHu_worker(session, task, save_path) for task in batch_tasks_data]
                batch_results = await asyncio.gather(*async_tasks_batch)

                async with file_lock:
                    with open(save_path, "a", encoding="utf-8") as f:
                        for res in batch_results:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

                all_results.extend(batch_results)

                pbar.update(len(batch_tasks_data))

                if i + batch_size < len(tasks):
                    pbar.set_description(f"Batch complete. Sleeping for {sleep_interval}s...")
                    await asyncio.sleep(sleep_interval)
                    pbar.set_description("Processing ShuiHu API requests")

        success_count = sum(1 for r in all_results if r and r.get("status") == "success")
        failed_count = len(all_results) - success_count
        print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")


def process_HongLou(input_text, llm='gpt'):
    max_tokens = 1000

    if llm == "gpt-3.5-turbo" or llm == "gpt":
        api_key = load_secret()['openai']['secret']
        call_api = openrouter_gpt_api_0613
    elif llm == "qwen-plus":
        api_key = load_secret()['tongyi']['secret']
        call_api = qwen_plus_api
    elif llm == "deepseek-chat":
        api_key = load_secret()['deepseek']['secret']
        call_api = official_deepseek_api
    else:
        raise ValueError(f"Unsupported llm: {llm}")

    messages = [{"role": "user", "content": input_text}]
    print(f"[PROCESS-HongLou] Processing input: {input_text[:40]}...")

    response = call_api(messages, api_key, max_tokens=max_tokens)
    response_dict = response.to_dict()

    return response_dict

async def HongLou_worker(session: aiohttp.ClientSession, task: Dict, save_path: str):
    async with SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, process_HongLou, task['prompt']['prompt'], task['llm']
            )

            response["id"] = task['prompt']["id"]
            response["prompt"] = task['prompt']["prompt"]
            response["status"] = "success"

            return response

        except Exception as e:
            error_message = {
                "id": task['prompt']["id"],
                "status": "failed",
                "error": str(e),
                "prompt": task['prompt']
            }
            return error_message

async def HongLou_batch_jobs(tasks: List[Dict], save_path: str):
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    async with aiohttp.ClientSession() as session:
        with sync_tqdm(total=len(tasks), desc="Processing HongLou API requests") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch_tasks_data = tasks[i:i + batch_size]
                async_tasks_batch = [HongLou_worker(session, task, save_path) for task in batch_tasks_data]
                batch_results = await asyncio.gather(*async_tasks_batch)

                async with file_lock:
                    with open(save_path, "a", encoding="utf-8") as f:
                        for res in batch_results:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

                all_results.extend(batch_results)

                pbar.update(len(batch_tasks_data))

                if i + batch_size < len(tasks):
                    pbar.set_description(f"Batch complete. Sleeping for {sleep_interval}s...")
                    await asyncio.sleep(sleep_interval)
                    pbar.set_description("Processing HongLou API requests")

        success_count = sum(1 for r in all_results if r and r.get("status") == "success")
        failed_count = len(all_results) - success_count
        print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")


def process_Pantheon(input_text, llm='gpt'):
    max_tokens = 1000

    if llm == "gpt-3.5-turbo" or llm == "gpt":
        api_key = load_secret()['openai']['secret']
        call_api = openrouter_gpt_api_0613
    elif llm == "qwen-plus":
        api_key = load_secret()['tongyi']['secret']
        call_api = qwen_plus_api
    elif llm == "deepseek-chat":
        api_key = load_secret()['deepseek']['secret']
        call_api = official_deepseek_api
    else:
        raise ValueError(f"Unsupported llm: {llm}")

    messages = [{"role": "user", "content": input_text}]
    print(f"[PROCESS-Pantheon] Processing input: {input_text[:40]}...")

    response = call_api(messages, api_key, max_tokens=max_tokens)
    response_dict = response.to_dict()

    return response_dict


async def Pantheon_worker(session: aiohttp.ClientSession, task: Dict, save_path: str):
    async with SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, process_Pantheon, task['prompt']['prompt'], task['llm']
            )

            response["id"] = task['prompt']["id"]
            response["prompt"] = task['prompt']["prompt"]
            response["status"] = "success"

            return response

        except Exception as e:
            error_message = {
                "id": task['prompt']["id"],
                "status": "failed",
                "error": str(e),
                "prompt": task['prompt']
            }
            return error_message


async def Pantheon_batch_jobs(tasks: List[Dict], save_path: str):
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    async with aiohttp.ClientSession() as session:
        with sync_tqdm(total=len(tasks), desc="Processing Pantheon API requests") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch_tasks_data = tasks[i:i + batch_size]
                async_tasks_batch = [Pantheon_worker(session, task, save_path) for task in batch_tasks_data]
                batch_results = await asyncio.gather(*async_tasks_batch)

                async with file_lock:
                    with open(save_path, "a", encoding="utf-8") as f:
                        for res in batch_results:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

                all_results.extend(batch_results)

                pbar.update(len(batch_tasks_data))

        success_count = sum(1 for r in all_results if r and r.get("status") == "success")
        failed_count = len(all_results) - success_count
        print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")


def process_Pantheon_Geo(input_text, llm='gpt'):
    max_tokens = 1000

    if llm == "gpt-3.5-turbo" or llm == "gpt":
        api_key = load_secret()['openai']['secret']
        call_api = openrouter_gpt_api_0613
    elif llm == "qwen-plus":
        api_key = load_secret()['tongyi']['secret']
        call_api = qwen_plus_api
    elif llm == "deepseek-chat":
        api_key = load_secret()['deepseek']['secret']
        call_api = official_deepseek_api
    else:
        raise ValueError(f"Unsupported llm: {llm}")

    messages = [{"role": "user", "content": input_text}]
    print(f"[PROCESS-Pantheon-Geo] Processing input: {input_text[:40]}...")

    response = call_api(messages, api_key, max_tokens=max_tokens)
    response_dict = response.to_dict()

    return response_dict


async def Pantheon_Geo_worker(session: aiohttp.ClientSession, task: Dict, save_path: str):
    async with SEMAPHORE:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, process_Pantheon_Geo, task['prompt'], task['llm']
            )

            response["id"] = task["id"]
            response["prompt"] = task["prompt"]
            response["status"] = "success"
            response["Occupation"] = task.get("Occupation", "")

            return response

        except Exception as e:
            error_message = {
                "id": task["id"],
                "status": "failed",
                "error": str(e),
                "prompt": task["prompt"],
                "Occupation": task.get("Occupation", "")
            }
            return error_message


async def run_Pantheon_Geo_batch_jobs(tasks: List[Dict], save_path: str):
    all_results = []
    batch_size = CONCURRENCY_LIMIT
    sleep_interval = 3

    async with aiohttp.ClientSession() as session:
        with sync_tqdm(total=len(tasks), desc="Processing Pantheon Geo API requests") as pbar:
            for i in range(0, len(tasks), batch_size):
                batch_tasks_data = tasks[i:i + batch_size]
                async_tasks_batch = [Pantheon_Geo_worker(session, task, save_path) for task in batch_tasks_data]
                batch_results = await asyncio.gather(*async_tasks_batch)

                async with file_lock:
                    with open(save_path, "a", encoding="utf-8") as f:
                        for res in batch_results:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")

                all_results.extend(batch_results)

                pbar.update(len(batch_tasks_data))

                if i + batch_size < len(tasks):
                    pbar.set_description(f"Batch complete. Sleeping for {sleep_interval}s...")
                    await asyncio.sleep(sleep_interval)
                    pbar.set_description("Processing Pantheon Geo API requests")

        success_count = sum(1 for r in all_results if r and r.get("status") == "success")
        failed_count = len(all_results) - success_count
        print(f"\nAPI processing finished. Success: {success_count}, Failed: {failed_count}")