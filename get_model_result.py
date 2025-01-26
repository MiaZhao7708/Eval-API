import os
import pandas
import json
import sys
import multiprocessing
import time
from model import get_model, thread_parallel
from utils import utils


def get_model_result(evaluation_set, model_name, result_dir='output/results'):
    model = get_model(model_name)
    result_file = os.path.join(result_dir, evaluation_set, model.name + '.json')
    eval_data = utils.read_data_file(evaluation_set)
    if os.path.exists(result_file):
        cache_result = json.load(open(result_file))
        evaluation_set = [
            cache_result.get(data['Prompt'], data)
            for data in evaluation_set
        ]
    else:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

    model_result, total_count, failure_count = {}, 0, 0
    for result in thread_parallel(model.process_data, evaluation_set, 20, name=model.name):
        if result.get('model', ''):
            total_count += 1
            model_result[result['Prompt']] = result
        else:
            failure_count += 1
            total_count += 1
            print(f'Access model failed. Failed Rate={failure_count}/{total_count}')

    with open(result_file, 'w') as f:
        json.dump(model_result, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) == 3:
        validate_set = sys.argv[1]
        model_names = sys.argv[2].split(',')

    tasks = []
    for model_name in model_names:
        tasks.append([validate_set, model_name])

    with multiprocessing.Pool(processes=len(model_names)) as pool:
        pool.starmap(get_model_result, tasks)

    end_time = time.time()
    run_time = end_time - start_time
    print(f"get_model_result cost {run_time} s")
