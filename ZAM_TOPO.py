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
from eval.benchmarks.Topo4MEC.scenario import ZAM_TOPO_Scenario as Scenario
from eval.metrics.metrics import SuccessRate, AvgLatency  # metric
from policies.demo.demo_greedy import GreedyPolicy
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

def error_handler_3(error: Exception):
    print(3, error)

def error_handler_4(error: Exception, arrival_times, arrival_pointer, task_timers, now):
    _, _, task_id = error.args[0]
    # Increament the arrival_pointer till the generated time[pointer] is greater than the current time
    node = task_timers[task_id]

    while arrival_pointer[node] < len(arrival_times[node]) and arrival_times[node][arrival_pointer[node]] <= now + 2:
        arrival_pointer[node] += 1

def main():
    flag = '25N50E'
    # flag = 'Tuple50K'
    # flag = 'Tuple100K'
    
    # Create the environment with the specified scenario and configuration files.
    scenario=Scenario(config_file=f"eval/benchmarks/Topo4MEC/data/{flag}/config.json", flag=flag)
    env = ZAM_env(scenario, config_file="core/configs/env_config.json")

    # Load the test dataset.
    data = pd.read_csv(f"eval/benchmarks/Topo4MEC/data/{flag}/testset.csv")
    time_slice = 500
    # Init the policy.
    policy = RoundRobinPolicy()
    # policy = GreedyPolicy()
    # Begin the simulation.
    arrival_times = {node_name: [] for node_name in env.scenario.node_id2name.values()}
    arrival_pointer = {node_name: 0 for node_name in env.scenario.node_id2name.values()}
    task_assign = {}  # Maps task_id to destination node name
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
    while True:
            # Process completed task information.
            while env.done_task_info:
                item = env.done_task_info.pop(0)
                # (Additional processing of completed tasks can be added here.)
            
            # Trust-based simulation updates.
            try:
                env.computeQoS()
            except Exception as e:
                error_handler_1(e)
            try:
                env.compute_trust()
            except Exception as e:
                error_handler_2(e)
            try:
                env.toggle_status(arrival_times, arrival_pointer)
            except Exception as e:
                error_handler_3(e)
            try:
                env.ballot_stuffing_attack()
            except Exception as e:
                print("Ballot Stuffing Attack error:", e)
            
            # Check if it's time to offload the task.
            if env.now >= generated_time:
                dst_id = policy.act(env, task)  # Offloading decision based on policy.
                dst_name = env.scenario.node_id2name[dst_id]
                env.process(task=task, dst_name=dst_name)
                launched_task_cnt += 1
                # Record trust simulation variables.
                arrival_times[dst_name].append(generated_time)
                task_assign[task_info['TaskID']] = dst_name
                break
            
            # Run the simulation for a time slice.
            try:
                env.run(until=until)
            except Exception as e:
                error_handler_4(e, arrival_times, arrival_pointer, task_assign, until)
            
            until += 1
            # Optionally add a small sleep to simulate real-time progression.
            time.sleep(0.0)

    # Continue the simulation until the last task successes/fails.
    while env.task_count < launched_task_cnt:
        try:
            env.computeQoS()
        except Exception as e:
            error_handler_1(e)
        try:
            env.compute_trust()
        except Exception as e:
            error_handler_2(e)
        try:
            env.toggle_status(arrival_times, arrival_pointer)
        except Exception as e:
            error_handler_3(e)
        try:
            env.run(until=until)
        except Exception as e:
            error_handler_4(e, arrival_times, arrival_pointer, task_assign, until)
        until += 1


    # Evaluation Metrics.
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
    
    # Stats Visualization for policy-based offloading.
    vis = VisStats(path_dir)
    vis.vis(env)

    # ------------------ Trust-based Visualization ------------------
    # Plot trust values of nodes over time.
    plt.figure(figsize=(10, 6))
    # Assuming env.trust_values is a dict mapping node IDs (or names) to a list of trust values.
    nodes_to_plot = list(env.trust_values.keys()) if hasattr(env, 'trust_values') else []
    malicious_nodes = []  # Specify malicious node ids if applicable.
    for node in nodes_to_plot:
        if node in malicious_nodes:
            plt.plot(env.trust_values[node][:time_slice], label=f'Node n{node}', linewidth=3)
        else:
            plt.plot(env.trust_values[node][:time_slice], label=f'Node n{node}', linewidth=1)
            
    # Prepare lists to collect attack marker coordinates.
    bsa_x, bsa_y = [], []      
    onoff_x, onoff_y = [], []     
    if hasattr(env, 'attacks'):
        for attack_time, events in env.attacks.items():
            if attack_time < 0 or attack_time > time_slice:
                continue
            for event in events:
                attacking_node = event.get("attacking_node", "")
                attack_type = event.get("attack_type", "")
                try:
                    node_index = int(attacking_node.strip('n'))
                except Exception as e:
                    continue
                time_index = int(attack_time)
                if time_index < len(env.trust_values.get(node_index, [])):
                    y_value = env.trust_values[node_index][time_index]
                    if attack_type == "ballot stuffing":
                        bsa_x.append(time_index)
                        bsa_y.append(y_value)
                    elif attack_type == "on-off attack":
                        onoff_x.append(time_index)
                        onoff_y.append(y_value)
    if bsa_x:
        plt.scatter(bsa_x, bsa_y, color='blue', marker='x', s=100, label='BSA Attack')
    if onoff_x:
        plt.scatter(onoff_x, onoff_y, color='red', marker='x', s=100, label='On-off Attack')

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trust Value', fontsize=12)
    plt.title(f'Trust Values of Nodes Over Time (Time 0-{time_slice})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, time_slice)
    plt.tight_layout()
    plt.show()
    
    # ------------------ Z-Score Detections Visualization ------------------
    plt.figure(figsize=(10, 6))
    zscore_x, zscore_y = [], []
    if hasattr(env, 'zscore_detections'):
        with open("detections_zscore.txt", "w") as f:
            f.write("Z_SCORE DETECTIONS\n")
            for detection_time, node_ids in env.zscore_detections.items():
                time_index = int(detection_time)
                valid_nodes = []
                for node_id in node_ids:
                    if time_index < len(env.trust_values.get(node_id, [])):
                        valid_nodes.append(str(node_id))
                        zscore_x.append(time_index)
                        zscore_y.append(env.trust_values[node_id][time_index])
                if valid_nodes:
                    f.write(f"Time: {time_index}, Nodes: {','.join(valid_nodes)}\n")
    if zscore_x:
        plt.scatter(zscore_x, zscore_y, color='green', marker='^', s=100, label='Z-Score Detection')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trust Value', fontsize=12)
    plt.title('Z-Score Malicious Detections', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, time_slice)
    plt.tight_layout()
    plt.show()
    
    # ------------------ Boxplot Detections Visualization ------------------
    plt.figure(figsize=(10, 6))
    boxplot_x, boxplot_y = [], []
    if hasattr(env, 'boxplot_detections'):
        with open("detections_boxplot.txt", "w") as f:
            f.write("BOXPLOT DETECTIONS\n")
            for detection_time, node_ids in env.boxplot_detections.items():
                time_index = int(detection_time)
                valid_nodes = []
                for node_id in node_ids:
                    if time_index < len(env.trust_values.get(node_id, [])):
                        valid_nodes.append(str(node_id))
                        boxplot_x.append(time_index)
                        boxplot_y.append(env.trust_values[node_id][time_index])
                if valid_nodes:
                    f.write(f"Time: {time_index}, Nodes: {','.join(valid_nodes)}\n")
    if boxplot_x:
        plt.scatter(boxplot_x, boxplot_y, color='purple', marker='D', s=100, label='Boxplot Detection')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trust Value', fontsize=12)
    plt.title('Boxplot Malicious Detections', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, time_slice)
    plt.tight_layout()
    plt.show()
    
    # ------------------ Detections Over Time ------------------
    # Z-Score detections count over time.
    if hasattr(env, 'zscore_detections'):
        zscore_times = sorted(env.zscore_detections.keys())
        zscore_counts = [len(env.zscore_detections[t]) for t in zscore_times]
        plt.figure(figsize=(10, 6))
        plt.plot(zscore_times, zscore_counts, marker='o', linestyle='-', color='green', label='Z-Score Detection Count')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Malicious Nodes Detected', fontsize=12)
        plt.title('Z-Score Malicious Detections Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    
    # Boxplot detections count over time.
    if hasattr(env, 'boxplot_detections'):
        boxplot_times = sorted(env.boxplot_detections.keys())
        boxplot_counts = [len(env.boxplot_detections[t]) for t in boxplot_times]
        plt.figure(figsize=(10, 6))
        plt.plot(boxplot_times, boxplot_counts, marker='o', linestyle='-', color='purple', label='Boxplot Detection Count')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Number of Malicious Nodes Detected', fontsize=12)
        plt.title('Boxplot Malicious Detections Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    
    # Visualization: frames to video (if supported by your vis module)
    try:
        vis_frame2video(env)
    except Exception as e:
        print("Error generating video from frames:", e)


if __name__ == '__main__':
    main()