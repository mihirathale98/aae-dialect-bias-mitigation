from evalplus.data import write_jsonl
import json
import os
mbpp_plus_task_ids = json.load(open('../data/redial/redial_gold/mbpp_plus_task_ids.json', 'r'))

def add_dummy_task_ids(eval_path,
                       mbpp_plus_task_ids):
    with open(eval_path, 'r') as f:
        to_eval = [json.loads(l) for l in f]
    to_eval_task_ids = set([problem['task_id'] for problem in to_eval])
    for task_id in mbpp_plus_task_ids:
        if task_id not in to_eval_task_ids:
            to_eval.append({'task_id':task_id, 'solution':''})

    # write back to the file
    write_jsonl(eval_path, to_eval)


def list_files(directory):
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            res.append(os.path.join(root, file))
    
    return res

def sanitize_files(alg_dir,
                   mbpp_plus_task_ids):

    # iterate and list all files in this directory or its subdirectories
    all_files = list_files(alg_dir)
    for file_path in all_files:
        if 'jsonl' not in file_path:
            continue
        # Check if the file ends with '_new.jsonl'
        if os.path.isfile(file_path) and 'to_eval' in file_path and 'sanitized' not in file_path:
            if file_path.replace('.jsonl', '-sanitized.jsonl') in all_files:
                continue
            
            # phi output unparsed format is weird so skip it
            if 'phi' in file_path and 'unparsed' in file_path:
                continue
            # Execute the command
            if 'mbpp' in file_path:
                # read jsonl
                with open(file_path, 'r') as f:
                    if len([json.loads(l) for l in f]) != 150:
                        continue
            else:
                with open(file_path, 'r') as f:
                    if len([json.loads(l) for l in f]) != 164:
                        continue
            
            # continue if the sanitized file already exists
            if os.path.exists(file_path.replace('.jsonl', '-sanitized.jsonl')):
                continue
            os.system(f'evalplus.sanitize --samples "{file_path}"')
            if 'mbpp' in file_path:
                add_dummy_task_ids(file_path.replace('.jsonl', '-sanitized.jsonl'), mbpp_plus_task_ids)    


import subprocess
import concurrent.futures

def run_subprocess(file_path):
    # try:
        # if os.path.isfile(file) and '-sanitized' in file:
    if 'humaneval' in file_path:
        data_name = 'humaneval'
    else:
        data_name = 'mbpp'
    # get the parent directory of the file
    parent_dir_path = os.path.dirname(file_path)
    # get the filename
    file = os.path.basename(file_path)
    print(parent_dir_path, file, data_name)
    absolute_path = os.path.abspath(parent_dir_path)
    print(absolute_path)
    result = subprocess.run(
        ['docker', 'run', '-v', f'{absolute_path}:/app', 'ganler/evalplus:v0.2.1', '--dataset', data_name, '--samples', file],
        capture_output=True,
        text=True
    )
    # # Print the captured output
    # with open(file_path.replace('.jsonl', '-sanitized_eval_results.json'), 'w') as f:
    #     f.write(result.stdout)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def test_code(alg_dir,
              max_workers=4):
    all_files = list_files(alg_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_path in all_files:
            print(file_path)
            if 'jsonl' not in file_path:
                continue
            if '-sanitized' in file_path and 'eval_results' not in file_path and file_path.replace('-sanitized.jsonl', '-sanitized_eval_results.json') not in all_files:
                future = executor.submit(run_subprocess, file_path)
                futures.append(future)
                # Wait for the subprocess to complete before clearing the Docker cache
                future.add_done_callback(lambda fut: executor.submit(subprocess.run, ['docker', 'system', 'prune', '-f'], capture_output=True, text=True))
        for future in concurrent.futures.as_completed(futures):
            future.result()  # To raise any exceptions that occurred


def sanitize_and_test(alg_dir, mbpp_plus_task_ids=mbpp_plus_task_ids, max_workers=8):
    sanitize_files(alg_dir, mbpp_plus_task_ids)
    test_code(alg_dir, max_workers=max_workers)


if __name__ == '__main__':
    sanitize_and_test("../outputs/algorithm")