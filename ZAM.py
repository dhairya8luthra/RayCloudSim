"""
Example simulation with additional trust metric
"""
import os
import sys
import time

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

def error_handler_4(error: Exception):
    print(4)

def main():
    # Create the Env
    scenario=Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    env = ZAM_env(scenario, config_file="core/configs/env_config.json")


    # Load simulated tasks
    data = pd.read_csv("examples/dataset/demo3_dataset.csv")
    simulated_tasks = list(data.iloc[:].values)
    n_tasks = len(simulated_tasks)

    # Check the arrival times of tasks for each node
    arrival_times = {node.name: [] for _, node in env.scenario.get_nodes().items()}
    arrival_pointer = {node.name: 0 for _, node in env.scenario.get_nodes().items()}


    # The Task are already sorted by generation time
    for task_info in simulated_tasks:
        arrival_times[task_info[8]].append(task_info[1])

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
                env.computeQoS()
            except Exception as e:
                error_handler_3(e)

            try:
                env.run(until=until)
            except Exception as e:
                error_handler_4(e)

            until += 1

        time.sleep(0.1)

    # Continue the simulation until the last task successes/fails.
    while env.process_task_cnt < len(simulated_tasks):
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
            env.computeQoS()
        except Exception as e:
            error_handler_3(e)

        try:
            env.run(until=until)
        except Exception as e:
            error_handler_4(e)

    env.close()

    # Visualization: frames to video
    vis_frame2video(env)


if __name__ == '__main__':
    main()