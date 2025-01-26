import json
import glob
import os
import sys
import time
import multiprocessing
from model import get_model, thread_parallel


def load_benchmark():
    dataset = []
    for fname in glob.glob('data/bench/*.jsonl'):
        for line in open(fname):
            data = json.loads(line.strip())
            dataset.append(data)
    print(f'Load #{len(dataset)} examples', flush=True)
    return dataset    


def get_bench_result(model_name, result_dir='output/results'):
    model = get_model(model_name)
    result_file = os.path.join(result_dir, 'bench', model.name + '.json')
    benchmark = load_benchmark()

    if os.path.exists(result_file):
        cache_result = {data['id']: data for data in json.load(open(result_file))}
        benchmark = [
            cache_result.get(data['id'], data)
            for data in benchmark
        ]
    else:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

    model_result, total_count, failure_count = [], 0
    for result in thread_parallel(model.process_data, benchmark, 20, name=model.name):
        if all([
            'model' in message
            for message in result['messages']
            if message['role'] == 'assistant'
        ]):
            total_count += 1
            model_result.append(result)
        else:
            total_count += 1
            failure_count += 1
            print(f'Access model failed. Failed Rate={failure_count}/{total_count}')

    with open(result_file, 'w') as f:
        json.dump(model_result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) == 2:
        model_names = sys.argv[1].split(',')

    tasks = []
    for model_name in model_names:
        tasks.append([model_name])

    with multiprocessing.Pool(processes=len(model_names)) as pool:
        pool.starmap(get_bench_result, tasks)

    end_time = time.time()
    run_time = end_time - start_time
    print(f"get_model_result cost {run_time} s")
