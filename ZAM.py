"""
Example simulation with additional trust metric
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

from examples.scenarios.zam_scenario import Scenario


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
    # Create the Env
    scenario=Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    env = ZAM_env(scenario, config_file="core/configs/env_config.json")

    time_slice = 500
    # Load simulated tasks
    data = pd.read_csv("examples/dataset/demo3_dataset.csv")
    simulated_tasks = list(data.iloc[:].values)
    n_tasks = len(simulated_tasks)

    # Check the arrival times of tasks for each node
    # Check the arrival times of tasks for each node
    arrival_times = {node.name: [] for _, node in env.scenario.get_nodes().items()}
    task_assign = {}
    arrival_pointer = {node.name: 0 for _, node in env.scenario.get_nodes().items()}

    # The Task are already sorted by generation time
    for task_info in simulated_tasks:
        arrival_times[task_info[8]].append(task_info[1])
        task_assign[task_info[2]] = task_info[8]

    # Begin Simulation
    until = 1
    for task_info in simulated_tasks:
        # header = ['TaskName', 'GenerationTime', 'TaskID', 'TaskSize', 'CyclesPerBit', 
        #           'TransBitRate', 'DDL', 'SrcName', 'DstName']
        generated_time, dst_name = task_info[1], task_info[8]
        task = Task(task_id=task_info[2],
                    task_size=task_info[3],
                    cycles_per_bit=task_info[4],
                    trans_bit_rate=task_info[5],
                    ddl=task_info[6],
                    src_name=task_info[7],
                    task_name=task_info[0]
                    )
        
        while True:
            # Catch the returned info of completed tasks
            while env.done_task_info:
                item = env.done_task_info.pop(0)
                # print(f"[{item[0]}]: {item[1:]}")

            if env.now == generated_time:
                env.process(task=task, dst_name=dst_name)
                break

            # Execute the simulation with error handler
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
                print(e)    

            try:
                env.run(until=until)
            except Exception as e:
                error_handler_4(e, arrival_times, arrival_pointer, task_assign, until)

            until += 1
        
        time.sleep(0.0)


    # Continue the simulation until the last task successes/fails.
    while env.task_count < len(simulated_tasks):
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
    env.close()

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
    nodes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    malicious_nodes = [2,4]

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
    
    # ---------------------------
    
    plt.figure(figsize=(10, 6))
    zscore_x, zscore_y = [], []
    zscore_x, zscore_y = [], []
    boxplot_x, boxplot_y = [], []
    
    # Open file to store the detections with one row per timestamp (Z-Score)
    with open("detections_zscore.txt", "w") as f:
        f.write("Z_SCORE DETECTIONS\n")
        for detection_time, node_ids in env.zscore_detections.items():
            time_index = int(detection_time)
            valid_nodes = []
            for node_id in node_ids:
                if time_index < len(env.trust_values[node_id]):
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
    
    # ---------------------------
    # Graph 3: Boxplot Malicious Detections
    plt.figure(figsize=(10, 6))
    with open("detections_boxplot.txt", "w") as f:
        f.write("BOXPLOT DETECTIONS\n")
        for detection_time, node_ids in env.boxplot_detections.items():
            time_index = int(detection_time)
            valid_nodes = []
            for node_id in node_ids:
                if time_index < len(env.trust_values[node_id]):
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
# --- Z-Score Detections Over Time ---
# Get sorted times for which a detection occurred
    zscore_times = sorted(env.zscore_detections.keys())
    zscore_counts = [len(env.zscore_detections[t]) for t in zscore_times]

    plt.figure(figsize=(10, 6))
    plt.plot(zscore_times, zscore_counts, marker='o', linestyle='-', color='green', label='Z-Score Detection Count')
    for t in zscore_times:
    # Get list of detected node IDs at time t and convert them to names like "n<id>"
        node_ids = env.zscore_detections[t]
        node_names = [f"n{node_id}" for node_id in node_ids]
        annotation_text = ', '.join(node_names)
    # Annotate above the point
       
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Malicious Nodes Detected', fontsize=12)
    plt.title('Z-Score Malicious Detections Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# --- Boxplot Detections Over Time ---
    boxplot_times = sorted(env.boxplot_detections.keys())
    boxplot_counts = [len(env.boxplot_detections[t]) for t in boxplot_times]

    plt.figure(figsize=(10, 6))
    plt.plot(boxplot_times, boxplot_counts, marker='o', linestyle='-', color='purple', label='Boxplot Detection Count')
    for t in boxplot_times:
        node_ids = env.boxplot_detections[t]
        node_names = [f"n{node_id}" for node_id in node_ids]
        annotation_text = ', '.join(node_names)
        
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Number of Malicious Nodes Detected', fontsize=12)
    plt.title('Boxplot Malicious Detections Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Visualization: frames to video
    vis_frame2video(env)

if __name__ == '__main__':
    main()