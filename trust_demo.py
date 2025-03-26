"""
Example simulation with additional trust metric
"""
import os
import sys
import time
from sklearn.metrics.pairwise import cosine_similarity

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd

from core.env import Env_Trust
from core.task import Task
from core.vis import *

from examples.scenarios.trust_scenario_1 import Scenario


def error_handler_1(error: Exception):
    print(1, error)

def error_handler_2(error: Exception):
    print(2, error)
    exit()

def error_handler_3(error: Exception, arrival_times, arrival_pointer, task_timers, now):
    _, _, task_id = error.args[0]
    # Increament the arrival_pointer till the generated time[pointer] is greater than the current time
    node = task_timers[task_id]

    while arrival_pointer[node] < len(arrival_times[node]) and arrival_times[node][arrival_pointer[node]] <= now + 2:
        arrival_pointer[node] += 1

def error_handler_4(error: Exception):
    print(4, error)
    exit()

def main():
    # Create the Env
    scenario=Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    env = Env_Trust(scenario, config_file="core/configs/env_config.json")


    # Load simulated tasks
    data = pd.read_csv("examples/dataset/task_dataset.csv")
    simulated_tasks = list(data.iloc[:].values)
    n_tasks = len(simulated_tasks)

    # Check the arrival times of tasks for each node
    arrival_times = {node.name: [] for _, node in env.scenario.get_nodes().items()}
    task_assign = {}
    arrival_pointer = {node.name: 0 for _, node in env.scenario.get_nodes().items()}

    node_embedding = env.generate_static_embeddings().iloc[1:, 1:].to_numpy()
    similarity_matrix = cosine_similarity(node_embedding)

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
                env.compute_trust()
            except Exception as e:
                error_handler_1(e)

            try:
                env.toggle_status(arrival_times, arrival_pointer)
            except Exception as e:
                error_handler_2(e)

            try:
                env.run(until=until)
            except Exception as e:
                error_handler_3(e, arrival_times, arrival_pointer, task_assign, until)

            try:
                env.generate_spatial_embeddings()
            except Exception as e:
                error_handler_4(e)

            print(arrival_times['n1'], arrival_pointer['n1'], env.now, len(env.scenario.get_node('n1').task_buffer.task_ids[:]))
            until += 1
        # time.sleep(0.2)

    # Continue the simulation until the last task successes/fails.
    while env.task_count < len(simulated_tasks):
        until += 1
        try:
            env.compute_trust()
        except Exception as e:
            error_handler_1(e)

        try:
            env.toggle_status(arrival_times, arrival_pointer)
        except Exception as e:
            error_handler_2(e)

        try:
            env.run(until=until)
        except Exception as e:
            error_handler_3(e, arrival_times, arrival_pointer, task_assign, until)

        try:
            env.generate_spatial_embeddings()
        except Exception as e:
            error_handler_4(e)

    env.close()

    # Visualization: frames to video
    vis_frame2video(env)
    print("Similarity Matrix", similarity_matrix)

if __name__ == '__main__':
    main()