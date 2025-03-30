"""
This script demonstrates how to tune hyperparameters on the Pakistan dataset sequentially.
"""

import os
import sys
import time
import itertools
import pandas as pd
import numpy as np

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.env import ZAM_env
from core.task import Task
from examples.scenarios.zam_scenario import Scenario
from policies.demo.demo_round_robin import RoundRobinPolicy

def run_simulation(hyperparams):
    """
    Runs the simulation with the given hyperparameters and returns evaluation metrics.
    hyperparams is a dict that can include: OLD_WEIGHT, THRESHOLD, lambda_task, lambda_time, lambda_base, ALPHA, BETA.
    """
    scenario = Scenario(config_file="eval/benchmarks/Topo4MEC/data/25N50E/config.json")
    env = ZAM_env(
        scenario, 
        config_file="core/configs/env_config.json",
        OLD_WEIGHT=hyperparams.get('OLD_WEIGHT', 0.8),
        THRESHOLD=hyperparams.get('THRESHOLD', 1.2),
        lambda_task=hyperparams.get('lambda_task', 0.7),
        lambda_time=hyperparams.get('lambda_time', 0.3),
        lambda_base=hyperparams.get('lambda_base', 0.5),
        ALPHA=hyperparams.get('ALPHA', 0.7),
        BETA=hyperparams.get('BETA', 0.3)
    )

    data = pd.read_csv("eval/benchmarks/Topo4MEC/data/25N50E/testset.csv")
    simulated_tasks = list(data.iloc[:].values)
    policy = RoundRobinPolicy()

    arrival_times = {node.name: [] for _, node in env.scenario.get_nodes().items()}
    next_arrival = {node.name: 0 for _, node in env.scenario.get_nodes().items()}
    for task_info in simulated_tasks:
        arrival_times[task_info[7]].append(task_info[1])

    launched_task_cnt = 0
    until = 0

    # Main simulation loop (processing tasks from the dataset)
    for i, task_info in data.iterrows():
        generated_time = task_info['GenerationTime']
        task = Task(
            task_id=task_info['TaskID'],
            task_size=task_info['TaskSize'],
            cycles_per_bit=task_info['CyclesPerBit'],
            trans_bit_rate=task_info['TransBitRate'],
            ddl=task_info['DDL'],
            src_name=task_info['SrcName'],
            task_name=task_info['TaskName']
        )
        env.scenario.get_node(task_info['SrcName']).isBusy += 1

        while True:
            while env.done_task_info:
                env.done_task_info.pop(0)
            if env.now >= generated_time:
                dst_node = policy.act(env, task)
                env.process(task=task, dst_name=f'n{dst_node}')
                launched_task_cnt += 1
                break
            try:
                env.computeQoS()
            except Exception:
                pass
            try:
                env.compute_trust()
            except Exception:
                pass
            try:
                env.toggle_dynamic(arrival_times, next_arrival)
            except Exception:
                pass
            try:
                env.ballot_stuffing_attack()
            except Exception:
                pass
            try:
                env.run(until=until)
            except Exception:
                pass
            until += 1

    # Continue simulation until all tasks are processed
    while env.task_count < launched_task_cnt:
        until += 1
        try:
            env.computeQoS()
        except Exception:
            pass
        try:
            env.compute_trust()
        except Exception:
            pass
        try:
            env.toggle_dynamic(arrival_times, next_arrival)
        except Exception:
            pass
        try:
            env.ballot_stuffing_attack()
        except Exception:
            pass
        try:
            env.run(until=until)
        except Exception:
            pass

    # Evaluate using the confusion metrics from ZAM_env (using the Z-Score method)
    TP = env.true_positive
    TN = env.true_negative
    FP = env.false_positive
    FN = env.false_negative
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    env.close()
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def run_single_combination(params):
    metric = run_simulation(params)
    result = {
        'OLD_WEIGHT': params['OLD_WEIGHT'],
        'THRESHOLD': params['THRESHOLD'],
        'lambda_task': params['lambda_task'],
        'lambda_time': params['lambda_time'],
        'lambda_base': params['lambda_base'],
        'ALPHA': params['ALPHA'],
        'BETA': params['BETA'],
        'accuracy': metric['accuracy'],
        'precision': metric['precision'],
        'recall': metric['recall'],
        'f1': metric['f1']
    }
    return result

if __name__ == '__main__':
    # Define grid for independent hyperparameters: values from 0.1 to 1.0 (step 0.2)
    old_weight_vals   = np.arange(0.1, 1.1, 0.2).tolist()
    threshold_vals    = np.arange(0.1, 1.1, 0.2).tolist()
    lambda_task_vals  = np.arange(0.1, 1.1, 0.2).tolist()  # lambda_time will be 1 - lambda_task
    lambda_base_vals  = np.arange(0.1, 1.1, 0.2).tolist()
    alpha_vals        = np.arange(0.1, 1.1, 0.2).tolist()     # BETA will be 1 - ALPHA

    all_params = []
    for ow, th, lt, lb, al in itertools.product(old_weight_vals, threshold_vals, lambda_task_vals, lambda_base_vals, alpha_vals):
        params = {
            'OLD_WEIGHT': ow,
            'THRESHOLD': th,
            'lambda_task': lt,
            'lambda_time': 1 - lt,
            'lambda_base': lb,
            'ALPHA': al,
            'BETA': 1 - al
        }
        all_params.append(params)

    total_combinations = len(all_params)
    print(f"Total combinations to test: {total_combinations}")

    tuning_results = []
    for i, params in enumerate(all_params, start=1):
        print(f"Testing combination {i}/{total_combinations}: "
              f"OLD_WEIGHT={params['OLD_WEIGHT']}, THRESHOLD={params['THRESHOLD']}, "
              f"lambda_task={params['lambda_task']}, lambda_time={params['lambda_time']}, "
              f"lambda_base={params['lambda_base']}, ALPHA={params['ALPHA']}, BETA={params['BETA']}")
        result = run_single_combination(params)
        tuning_results.append(result)
        print(f"Result: {result}")
        if i % 10 == 0:
            print(f"Completed {i}/{total_combinations} combinations...")

    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.to_csv("logs/hyperparameter_tuning_results.csv", index=False)
    print("Hyperparameter tuning complete. CSV file has been saved.")
