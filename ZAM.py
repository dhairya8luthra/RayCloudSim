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

def error_handler_2(error: Exception):
    print(2, error)

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


    # Load simulated tasks
    data = pd.read_csv("examples/dataset/task_dataset.csv")
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
                env.run(until=until)
            except Exception as e:
                error_handler_4(e, arrival_times, arrival_pointer, task_assign, until)

            until += 1
        
        time.sleep(0.1)


    # Continue the simulation until the last task successes/fails.
    while env.process_task_cnt < len(simulated_tasks):
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

   

    nodes_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    malicious_nodes = [2,5,11]

    plt.figure(figsize=(10, 6))
    for node in nodes_to_plot:
        if node in malicious_nodes:
            plt.plot(env.trust_values[node][:300], label=f'Node n{node}', color='darkred', linewidth=3)
        else:
            plt.plot(env.trust_values[node][:300], label=f'Node n{node}', linewidth=1)
            
    # Prepare lists to collect attack marker coordinates.
    bsa_x, bsa_y = [], []      
    onoff_x, onoff_y = [], []     
    # Loop through the attacks dictionary to get marker positions.
    for attack_time, events in env.attacks.items():
    # Only consider attacks within time 0-300.
        if attack_time < 0 or attack_time > 300:
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
    plt.title('Trust Values of Nodes n1 and n12 Over Time (Time 0-300)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, 300)
    plt.tight_layout()
    plt.show()


    # Visualization: frames to video
    vis_frame2video(env)

if __name__ == '__main__':
    main()