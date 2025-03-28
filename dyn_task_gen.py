"""
This script demonstrates how to use the Pakistan dataset.
"""

import os
import sys
import time
import matplotlib.pyplot as plt
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from core.env import ZAM_env
from core.task import Task
from core.vis import *
from core.vis.vis_stats import VisStats
from examples.scenarios.zam_scenario import Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency  # metric
from policies.demo.demo_round_robin import RoundRobinPolicy



def create_log_dir(algo_name, **params):
    """Creates a directory for storing the training/testing metrics logs.

    Args:
        algo_name (str): The name of the algorithm.
        **params: Additional parameters to be included in the directory name.

    Returns:
        str: The path to the created log directory.
    """
    # Create the algorithm-specific directory if it doesn't exist
    algo_dir = f"logs/{algo_name}"
    if not os.path.exists(algo_dir):
        os.makedirs(algo_dir)

    # Build the parameterized part of the directory name
    params_str = ""
    for key, value in params.items():
        params_str += f"{key}_{value}_"
    index = 0  # Find an available directory index
    log_dir = f"{algo_dir}/{params_str}{index}"
    while os.path.exists(log_dir):
        index += 1
        log_dir = f"{algo_dir}/{params_str}{index}"
    
    # Create the final log directory
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir

def error_handler_1(error: Exception):
    print(1, error)
    exit()

def error_handler_2(error: Exception):
    print(2, error)
    exit()

def error_handler_3(error: Exception, until):
    print(3, error, until)
    exit()

def error_handler_4(error: Exception):
    print(4, error)
    exit()

def main():
    flag = 'Tuple30K'
    # flag = 'Tuple50K'
    # flag = 'Tuple100K'
    
    # Create the environment with the specified scenario and configuration files.
    scenario=Scenario(config_file=f"eval/benchmarks/Topo4MEC/data/25N50E/config.json")
    env = ZAM_env(scenario, config_file="core/configs/env_config.json")

    time_slice = 1000

    arrival_times = {node.name: [] for _, node in env.scenario.get_nodes().items()}
    next_arrival = {node.name: 0 for _, node in env.scenario.get_nodes().items()}

    # Load the test dataset.
    data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/25N50E/testset.csv")
    simulated_tasks = list(data.iloc[:].values)

    # Init the policy.
    policy = RoundRobinPolicy()

    for task_info in simulated_tasks:
        arrival_times[task_info[7]].append(task_info[1])

    # Begin the simulation.
    until = 0
    launched_task_cnt = 0
    path_dir = create_log_dir("vis/DemoGreedy", flag=flag)

    for i, task_info in data.iterrows():
        generated_time = task_info['GenerationTime']
        task = Task(task_id=task_info['TaskID'],
                    task_size=task_info['TaskSize'],
                    cycles_per_bit=task_info['CyclesPerBit'],
                    trans_bit_rate=task_info['TransBitRate'],
                    ddl=task_info['DDL'],
                    src_name=task_info['SrcName'],
                    task_name=task_info['TaskName'])

        env.scenario.get_node(task_info['SrcName']).isBusy += 1

        # Make the src node online if we need to.
        src_node = env.scenario.get_node(task_info['SrcName']).name
        # while next_arrival[src_node] < len(arrival_times[src_node]) and arrival_times[src_node][next_arrival[src_node]] <= env.controller.now + 2:
        #             next_arrival[src_node] += 1

        while True:
            # Catch completed task information.
            while env.done_task_info:
                item = env.done_task_info.pop(0)
            
            if env.now >= generated_time:
                dst_node = policy.act(env, task)  # offloading decision
                env.process(task=task, dst_name=f'n{dst_node}')
                launched_task_cnt += 1
                break

            try:
                env.computeQoS()
            except Exception as e:
                error_handler_1(e)

            try:
                env.compute_trust()
            except Exception as e:
                error_handler_2(e)

            try:
                env.toggle_dynamic(arrival_times, next_arrival)
            except Exception as e:
                error_handler_3(e, until=until)

            try:
                env.ballot_stuffing_attack()
            except Exception as e:
                print(e)  

            # Execute the simulation with error handler.
            try:
                env.run(until=until)
            except Exception as e:
                pass

            until += 1


    # Continue the simulation until the last task successes/fails.
    while env.task_count < launched_task_cnt:
        print(until)
        until += 1
        try:
            env.computeQoS()
        except Exception as e:
            error_handler_1(e)

        try:
            env.compute_trust()
        except Exception as e:
            error_handler_2(e)

        try:
            env.toggle_dynamic(arrival_times, next_arrival)
        except Exception as e:
            error_handler_3(e, until=until)

        try:
            env.ballot_stuffing_attack()
        except Exception as e:
            print(e)  

        # Execute the simulation with error handler.
        try:
            env.run(until=until)
        except Exception as e:
            pass

    # Evaluation
    print("\n===============================================")
    print("Evaluation:")
    print("===============================================\n")

    print("-----------------------------------------------")
    m1 = SuccessRate()
    r1 = m1.eval(env.logger.task_info)
    print(f"The success rate of all tasks: {r1:.4f}")
    print("-----------------------------------------------\n")

    print("-----------------------------------------------")
    m2 = AvgLatency()
    r2 = m2.eval(env.logger.task_info)
    print(f"The average latency per task: {r2:.4f}")

    print(f"The average energy consumption per node: {env.avg_node_energy():.4f}")
    print("-----------------------------------------------\n")

    env.close()
    
    # # Stats Visualization
    # vis = VisStats(path_dir)
    # vis.vis(env)

    # Print the confusion metrics - Z Score
    print("------------------------------------------------------")
    print("Z-Score Confusion Matrix:")
    print("True Positives:", env.true_positive)
    print("True Negatives:", env.true_negative)
    print("False Positives:", env.false_positive)
    print("False Negatives:", env.false_negative)
    print("Accuracy:", (env.true_positive + env.true_negative) / (env.true_positive + env.true_negative + env.false_positive + env.false_negative))
    print("Precision:", env.true_positive / (env.true_positive + env.false_positive))
    print("F1 Score:", (2 * env.true_positive) / (2 * env.true_positive + env.false_positive + env.false_negative))
    print("------------------------------------------------------\n")

    # Print the confusion metrics - Boxplot
    print("------------------------------------------------------")
    print("Boxplot Confusion Matrix:")
    print("True Positives:", env.true_positive_boxplot)
    print("True Negatives:", env.true_negative_boxplot)
    print("False Positives:", env.false_positive_boxplot)
    print("False Negatives:", env.false_negative_boxplot)
    print("Accuracy:", (env.true_positive_boxplot + env.true_negative_boxplot) / (env.true_positive_boxplot + env.true_negative_boxplot + env.false_positive_boxplot + env.false_negative_boxplot))
    print("Precision:", env.true_positive_boxplot / (env.true_positive_boxplot + env.false_positive_boxplot))
    print("F1 Score:", (2 * env.true_positive_boxplot) / (2 * env.true_positive_boxplot + env.false_positive_boxplot + env.false_negative_boxplot))
    print("------------------------------------------------------\n")

    nodes_to_plot = [i for i in range(len(env.scenario.get_nodes()))]
    malicious_nodes = [3, 4, 10, 12, 14, 21, 23]

    # vis_frame2video(env)

    plt.figure(figsize=(10, 6))
    for node in nodes_to_plot:
        if node in malicious_nodes:
            plt.plot(env.trust_values[node][:time_slice], label=f'Node n{node}', color='darkred', linewidth=3)
        else:
            plt.plot(env.trust_values[node][:time_slice], label=f'Node n{node}', linewidth=1)
            
    # Prepare lists to collect attack marker coordinates.
    bsa_x, bsa_y = [], []      
    onoff_x, onoff_y = [], []     
    # Loop through the attacks dictionary to get marker positions.
    for attack_time, events in env.attacks.items():
    # Only consider attacks within time 0-time_slice.
        if attack_time < 0 or attack_time > time_slice:
            continue
        for event in events:
            attacking_node = event["attacking_node"]
            attack_type = event["attack_type"]
            try:
                node_index = int(attacking_node.strip('n'))
            except Exception as e:
                continue
        # Use the simulation time as the x coordinate.
        time_index = int(attack_time)
        # Ensure that the trust value list is long enough.
        if time_index < len(env.trust_values[node_index]):
            y_value = env.trust_values[node_index][time_index]
            if attack_type == "ballot stuffing":
                bsa_x.append(time_index)
                bsa_y.append(y_value)
            elif attack_type == "on-off attack":
                onoff_x.append(time_index)
                onoff_y.append(y_value)

    # Plot attack markers if any.
    if bsa_x:
        plt.scatter(bsa_x, bsa_y, color='blue', marker='x', s=100, label='BSA Attack')
    if onoff_x:
        plt.scatter(onoff_x, onoff_y, color='red', marker='x', s=100, label='On-off Attack')

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trust Value', fontsize=12)
    plt.title(f'Trust Values of Nodes n1 and n12 Over Time (Time 0-{time_slice})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, time_slice)
    plt.tight_layout()
    plt.show()
   

if __name__ == '__main__':
    main()
