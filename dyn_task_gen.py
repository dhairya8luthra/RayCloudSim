import os
import sys
import random
import networkx as nx

from core.env import Env_Trust
from core.task import Task
from core.vis import *
from examples.scenarios.trust_scenario_1 import Scenario

def error_handler(error: Exception):
    pass

def main():
    scenario = Scenario(config_file="examples/scenarios/configs/trust_config_1.json")
    env = Env_Trust(scenario, config_file="core/configs/env_config.json")

    NUM_TASKS = 20
    NODES = ["n" + str(i) for i in range(13)]
    ONLINE_NODES = []
    ACTIVE_NODES = []
    until = 1
    
    env.up = {}
    env.down = {}
    
    for i in range(NUM_TASKS):
        up_nodes = random.sample([n for n in NODES if n not in ACTIVE_NODES], random.randint(1, 4))
        down_nodes = random.sample(ONLINE_NODES, random.randint(0, len(ONLINE_NODES)//2))
        
        env.up[until] = up_nodes
        env.down[until] = down_nodes
        
        for node in up_nodes:
            if node not in ONLINE_NODES:
                ONLINE_NODES.append(node)
        
        for node in down_nodes:
            if node in ONLINE_NODES:
                ONLINE_NODES.remove(node)
                if node in ACTIVE_NODES:
                    ACTIVE_NODES.remove(node)
                    env.process_task_cnt += len(env.task_queues.get(node, []))
                    env.task_queues[node] = []
        
        tasks = []
        for _ in range(random.randint(1, 3)):
            src = random.choice(ONLINE_NODES)
            dst = random.choice(ONLINE_NODES)
            if src == dst or not len(scenario.infrastructure.get_shortest_path(src_name=src, dst_name=dst)) == 0:
                continue
            task = Task(task_id=i,
                        task_size=random.randint(10, 100),
                        cycles_per_bit=random.randint(1, 10),
                        trans_bit_rate=random.randint(20, 100),
                        ddl=random.randint(50, 100),
                        src_name=src,
                        task_name=f"t{i}")
            tasks.append((task, dst))
        
        for task, dst in tasks:
            env.process(task=task, dst_name=dst)
        
        while env.done_task_info:
            item = env.done_task_info.pop(0)
        
        try:
            env.update_trust()
            env.toggle_status()
            env.run(until=until)
        except Exception as e:
            error_handler(e)
        
        until += 1
    
    while env.process_task_cnt < NUM_TASKS:
        print(f"Process task count: {env.process_task_cnt}")
        until += 1
        try:
            env.update_trust()
            env.toggle_status()
            env.run(until=until)
        except Exception as e:
            error_handler(e)
    
    env.close()
    vis_frame2video(env)

if __name__ == '__main__':
    main()
