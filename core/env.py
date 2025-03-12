import json
import os
import simpy
import numpy as np
import networkx as nx
import pandas as pd
from node2vec import Node2Vec

from typing import Optional, Tuple

from core.base_scenario import BaseScenario
from core.infrastructure import Link
from core.task import Task

from zoo.node import MaliciousNode, TrustNode, ZAMNode, ZAMMalicious

__all__ = ["EnvLogger", "Env", "Env_Trust"]


# Flags
FLAG_TASK_EXECUTION_DONE = 0
FLAG_TASK_EXECUTION_FAIL = 1

# Execution Flags
FLAG_TASK_EXECUTION_TIMEOUT = 2
FLAG_TASK_EXECUTION_NET_CONGESTION = 3
FLAG_TASK_EXECUTION_NO_PATH = 4
FLAG_TASK_INSUFFICIENT_BUFFER = 5
FLAG_TASK_DUPLICATE_ID = 6
FLAG_TASK_ISOLATED_WIRELESS_NODE = 7

"""
-> FAIL           => Reduce the trust
-> SUCCESS        => Increase the trust
-> TIMEOUT        => Reduce the trust a little
-> NO_PATH        => Don't change the trust
-> NET_CONGESTION => Don't change the trust
"""


def user_defined_info():
    """This is where user can define more infos for completed tasks."""
    return None


class EnvLogger:
    def __init__(self, controller, is_open=True):
        self.controller = controller
        self.is_open = is_open  # is_open=False can speed up training

        self.task_info = {}
        self.node_info = {}

    def log(self, content):
        if self.is_open:
            print("[{:.2f}]: {}".format(self.controller.now, content))

    def append(self, info_type, key, val):
        """Record key information during the simulation.

        Args:
            info_type: 'task' or 'node'
            key: task id or node id
            val: 
                if info_type == 'task': (code: int, info: list)
                    - 0 (success) || [*]
                    - 1 (failure) || [detailed error info,]
                if info_type == 'node':
                    - energy consumption
        """
        assert info_type in ['task', 'node']
        if info_type == 'task':
            self.task_info[key] = val
        else:
            self.node_info[key] = val

    def reset(self):
        self.task_info = {}
        self.node_info = {}


class Env:

    def __init__(self, scenario: BaseScenario, config_file):
        # Load the config file
        with open(config_file, 'r') as fr:
            self.config = json.load(fr)
        assert len(self.config['VisFrame']['TargetNodeList']) <= 13, \
            "For visualization layout considerations, the default number of tracked nodes " \
            "does not exceed ten, and users are permitted to modify the layout for extension."
        
        self.scenario = scenario
        self.controller = simpy.Environment()
        self.logger = EnvLogger(self.controller, is_open=True)

        self.active_task_dict = {}  # store current active tasks
        self.done_task_info = []  # catch infos of completed tasks
        self.done_task_collector = simpy.Store(self.controller)
        self.process_task_cnt = 0  # counter

        # self.processed_tasks = []  # for debug

        self.reset()

        # Launch the monitor process
        self.monitor_process = self.controller.process(
            self.monitor_on_done_task_collector())
        
        # Launch all energy recorder processes
        self.energy_recorders = {}
        for _, node in self.scenario.get_nodes().items():
            self.energy_recorders[node.node_id] = self.controller.process(self.energy_clock(node))

        # Launch the info recorder for frames
        if self.config['Basic']['VisFrame'] == "on":
            os.makedirs(self.config['VisFrame']['LogInfoPath'], exist_ok=True)
            os.makedirs(self.config['VisFrame']['LogFramesPath'], exist_ok=True)
            self.info4frame = {}
            self.info4frame_recorder = self.controller.process(self.info4frame_clock())

    @property
    def now(self):
        """The current simulation time."""
        return self.controller.now

    def run(self, until):
        """Run the simulation until the given criterion `until` is met."""
        self.controller.run(until)

    def reset(self):
        """Reset the Env."""
        # Interrupt all activate tasks
        for p in self.active_task_dict.values():
            if p.is_alive:
                p.interrupt()
        self.active_task_dict.clear()

        self.process_task_cnt = 0

        # Reset scenario state
        self.scenario.reset()

        # Reset the logger
        self.logger.reset()

        # Clear task monitor info
        self.done_task_collector.items.clear()
        del self.done_task_info[:]

    def process(self, **kwargs):
        """Must be called with keyword args."""
        task_generator = self.execute_task(**kwargs)
        self.controller.process(task_generator)

    def execute_task(self, task: Task, dst_name=None):
        """Transmission and Execution logics.

        dst_name=None means the task is popped from the waiting deque.
        """
        # DuplicateTaskIdError check
        if task.task_id in self.active_task_dict.keys():
            self.process_task_cnt += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               val=(1, ['DuplicateTaskIdError',]))
            # self.processed_tasks.append(task.task_id)
            log_info = f"**DuplicateTaskIdError: Task {{{task.task_id}}}** " \
                       f"new task (name {{{task.task_name}}}) with a " \
                       f"duplicate task id {{{task.task_id}}}."
            self.logger.log(log_info)
            raise AssertionError(
                ('DuplicateTaskIdError', log_info, task.task_id)
            )

        # Check whether the task is re-activated from queuing
        flag_reactive = True if dst_name is None else False

        if flag_reactive:
            dst = task.dst
        else:
            self.logger.log(f"Task {{{task.task_id}}} generated in "
                            f"Node {{{task.src_name}}}")
            dst = self.scenario.get_node(dst_name)

        if not flag_reactive:
            # Do task transmission, if necessary
            if dst_name != task.src_name:  # task transmission
                try:
                    links_in_path = self.scenario.infrastructure.\
                        get_shortest_links(task.src_name, dst_name)
                # NetworkXNoPathError check
                except nx.exception.NetworkXNoPath:
                    self.process_task_cnt += 1
                    self.logger.append(info_type='task', 
                                       key=task.task_id, 
                                       val=(1, ['NetworkXNoPathError',]))
                    # self.processed_tasks.append(task.task_id)
                    log_info = f"**NetworkXNoPathError: Task " \
                               f"{{{task.task_id}}}** Node {{{dst_name}}} " \
                               f"is inaccessible"
                    self.logger.log(log_info)
                    raise EnvironmentError(
                        ('NetworkXNoPathError', log_info, task.task_id)
                    )
                # IsolatedWirelessNode check
                except EnvironmentError as e:
                    message = e.args[0]
                    if message[0] == 'IsolatedWirelessNode':
                        self.process_task_cnt += 1
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           val=(1, ['IsolatedWirelessNode',]))
                        # self.processed_tasks.append(task.task_id)
                        log_info = f"**IsolatedWirelessNode"
                        self.logger.log(log_info)
                        raise e

                for link in links_in_path:
                    if isinstance(link, Link):
                        # NetCongestionError check
                        if link.free_bandwidth < task.trans_bit_rate:
                            self.process_task_cnt += 1
                            self.logger.append(info_type='task', 
                                               key=task.task_id, 
                                               val=(1, ['NetCongestionError',]))
                            # self.processed_tasks.append(task.task_id)
                            log_info = f"**NetCongestionError: Task " \
                                       f"{{{task.task_id}}}** network " \
                                       f"congestion Node {{{task.src_name}}} " \
                                       f"--> {{{dst_name}}}"
                            self.logger.log(log_info)
                            raise EnvironmentError(
                                ('NetCongestionError', log_info, task.task_id)
                            )

                task.trans_time = 0

                # ---- Customize the wired/wireless transmission mode here ----
                # wireless transmission:
                if isinstance(links_in_path[0], Tuple):
                    wireless_src_name, wired_dst_name = links_in_path[0]
                    # task.trans_time += func(task, wireless_src_name,
                    #                         wired_dst_name)  # TODO
                    task.trans_time += 0  # (currently only a toy model)
                    links_in_path = links_in_path[1:]
                if isinstance(links_in_path[-1], Tuple):
                    wired_src_name, wireless_dst_name = links_in_path[-1]
                    # task.trans_time += func(task, wired_src_name,
                    #                         wireless_dst_name)  # TODO
                    task.trans_time += 0  # (currently only a toy model)
                    links_in_path = links_in_path[:-1]

                # wired transmission:
                # 0. base latency
                trans_base_latency = 0
                for link in links_in_path:
                    trans_base_latency += link.base_latency
                task.trans_time += trans_base_latency
                # Multi-hop
                task.trans_time += (task.task_size / task.trans_bit_rate) * len(links_in_path)
                # -------------------------------------------------------------

                self.scenario.send_data_flow(task.trans_flow, links_in_path)

                try:
                    self.logger.log(f"Task {{{task.task_id}}}: "
                                    f"{{{task.src_name}}} --> {{{dst_name}}}")
                    yield self.controller.timeout(task.trans_time)
                    task.trans_flow.deallocate()
                    self.logger.log(f"Task {{{task.task_id}}} arrived "
                                    f"Node {{{dst_name}}} with "
                                    f"{{{task.trans_time:.2f}}}s")
                except simpy.Interrupt:
                    pass
            else:
                task.trans_time = 0  # To avoid task.trans_time = -1

        # Task execution
        if not dst.free_cpu_freq > 0:
            # InsufficientBufferError check
            try:
                task.allocate(self.now, dst, pre_allocate=True)
                dst.append_task(task)
                self.logger.log(f"Task {{{task.task_id}}} is buffered in "
                                f"Node {{{task.dst_name}}}")
                return
            except EnvironmentError as e:
                self.process_task_cnt += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   val=(1, ['InsufficientBufferError',]))
                # self.processed_tasks.append(task.task_id)
                self.logger.log(e.args[0][1])
                raise e

        # ------------ Customize the execution mode here ------------
        if flag_reactive:
            # TimeoutError check
            try:
                task.allocate(self.now)
            except EnvironmentError as e:  # TimeoutError
                self.process_task_cnt += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   val=(1, ['TimeoutError',]))
                # self.processed_tasks.append(task.task_id)
                self.logger.log(e.args[0][1])

                # Activate a queued task
                waiting_task = task.dst.pop_task()
                if waiting_task:
                    self.process(task=waiting_task)
                
                raise e
            
            self.logger.log(f"Task {{{task.task_id}}} re-actives in "
                            f"Node {{{task.dst_name}}}, "
                            f"waiting {{{(task.wait_time - task.trans_time):.2f}}}s")
        else:
            task.allocate(self.now, dst)
        # -----------------------------------------------------------

        # Mark the task as active (i.e., execution status) task
        self.active_task_dict[task.task_id] = task
        try:
            self.logger.log(f"Processing Task {{{task.task_id}}} in"
                            f" {{{task.dst_name}}}")
            yield self.controller.timeout(task.exe_time)
            self.done_task_collector.put(
                (task.task_id,
                 FLAG_TASK_EXECUTION_DONE,
                 [dst_name, user_defined_info()]))
        except simpy.Interrupt:
            pass

    def monitor_on_done_task_collector(self):
        """Keep watch on the done_task_collector."""
        while True:
            if len(self.done_task_collector.items) > 0:
                while len(self.done_task_collector.items) > 0:
                    task_id, flag, info = self.done_task_collector.get().value
                    self.done_task_info.append((self.now, task_id, flag, info))

                    if flag == FLAG_TASK_EXECUTION_DONE:
                        task = self.active_task_dict[task_id]

                        waiting_task = task.dst.pop_task()

                        self.logger.log(f"Task {{{task_id}}} accomplished in "
                                        f"Node {{{task.dst_name}}} with "
                                        f"{{{task.exe_time:.2f}}}s")
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           val=(0, [task.trans_time, task.wait_time, task.exe_time]))
                        task.deallocate()
                        del self.active_task_dict[task_id]
                        self.process_task_cnt += 1
                        # self.processed_tasks.append(task.task_id)

                        if waiting_task:
                            self.process(task=waiting_task)

                    else:
                        raise ValueError("Invalid flag!")
            else:
                self.done_task_info = []
                # self.logger.log("")  # turn on: log on every time slot

            yield self.controller.timeout(1)
    
    def energy_clock(self, node):
        """Recorder of node's energy consumption."""
        while True:
            node.energy_consumption += node.idle_energy_coef
            node.energy_consumption += node.exe_energy_coef * (
                node.max_cpu_freq - node.free_cpu_freq) ** 3
            yield self.controller.timeout(1)
    
    def info4frame_clock(self):
        """Recorder the info required for simulation frames."""
        while True:
            self.info4frame[self.now] = {
                'node': {k: item.quantify_cpu_freq() 
                         for k, item in self.scenario.get_nodes().items()},
                'edge': {str(k): item.quantify_bandwidth() 
                         for k, item in self.scenario.get_links().items()},
            }
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.info4frame[self.now]['target'] = {
                    item: [self.scenario.get_node(item).active_task_ids[:], 
                           self.scenario.get_node(item).task_buffer.task_ids[:]]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            yield self.controller.timeout(1)

    @property
    def n_active_tasks(self):
        """The number of current active tasks."""
        return len(self.active_task_dict)

    def status(self, node_name: Optional[str] = None,
               link_args: Optional[Tuple] = None):
        return self.scenario.status(node_name, link_args)
    
    def avg_node_energy(self, node_name_list=None):
        return self.scenario.avg_node_energy(node_name_list) / 1000000
    
    def node_energy(self, node_name):
        return self.scenario.node_energy(node_name) / 1000000

    def close(self):
        # Record nodes' energy consumption.
        for _, node in self.scenario.get_nodes().items():
            self.logger.append(info_type='node', key=node.node_id, val=node.energy_consumption)
        
        # Save the info4frame
        if self.config['Basic']['VisFrame'] == "on":
            info4frame_json_object = json.dumps(self.info4frame, indent=4)
            with open(f"{self.config['VisFrame']['LogInfoPath']}/info4frame.json", 'w+') as fw:
                fw.write(info4frame_json_object)

        # Terminate activate processes
        self.monitor_process.interrupt()
        for p in self.energy_recorders.values():
            if p.is_alive:
                p.interrupt()
        self.energy_recorders.clear()
        if self.config['Basic']['VisFrame'] == "on":
            self.info4frame_recorder.interrupt()

        self.logger.log("Simulation completed!")


class Env_Trust(Env):

    def __init__(self, scenario: BaseScenario, config_file):
        super().__init__(scenario, config_file)
        self.down = {}
        self.up = {}
        self.down.setdefault(self.controller.now, [])
        self.up.setdefault(self.controller.now, [])
        self.ONLINE_NODES = [node for k, node in self.scenario.get_nodes().items() if node.get_online()]
        self.ACTIVE_NODES = []
        self.trust_messages = []

    def generate_static_embeddings(self):
        """
        Generates Node2Vec embeddings for the given infrastructure graph.
        """
        G = self.scenario.infrastructure.graph

        if len(G.nodes) == 0:
            print("No online nodes available for embedding generation.")
            exit(1)
            return None

        node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=100, workers=4)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)

        embeddings_dict = {node: model.wv[node] for node in G.nodes()}

        # Convert embeddings to DataFrame
        all_embeddings = []
        for node, emb in embeddings_dict.items():
            all_embeddings.append([node] + emb.tolist())
        
        columns = ["node"] + [f"dim_{i}" for i in range(16)]
        df_embeddings = pd.DataFrame(all_embeddings, columns=columns)
        
        # Save embeddings to CSV
        df_embeddings.to_csv("sta_node2vec_embeddings.csv", index=False)
        print("Node2Vec embeddings have been saved to 'node2vec_embeddings.csv'.")
        
        return df_embeddings

    def generate_spatial_embeddings(self):
        """
        Generates Node2Vec embeddings for the given infrastructure graph while ignoring offline nodes.
        """
        G = self.scenario.infrastructure.graph
        
        # Filter out offline nodes using node.get_online()
        online_nodes = [node for node in G.nodes if G.nodes[node]["data"].get_online()]
        
        # Create a subgraph with only online nodes
        G_online = G.subgraph(online_nodes).copy()
        
        if len(G_online.nodes) == 0:
            print("No online nodes available for embedding generation.")
            return None
        
        # Generate Node2Vec model
        node2vec = Node2Vec(G_online, dimensions=16, walk_length=10, num_walks=100, workers=4)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        
        # Store embeddings
        embeddings_dict = {node: model.wv[node] for node in G_online.nodes()}
        
        # Convert embeddings to DataFrame
        all_embeddings = []
        for node, emb in embeddings_dict.items():
            all_embeddings.append([node] + emb.tolist())
        
        columns = ["node"] + [f"dim_{i}" for i in range(16)]
        df_embeddings = pd.DataFrame(all_embeddings, columns=columns)
        
        # Save embeddings to CSV
        df_embeddings.to_csv("dyn_node2vec_embeddings.csv", index=False)
        print("Node2Vec embeddings have been saved to 'node2vec_embeddings.csv'.")
        
        return df_embeddings



    def info4frame_clock(self):
        """Recorder the info required for simulation frames."""
        # Offline Node: 0.0
        # Online Malicious Node: 1.0
        # Online Non-Malicious Node: -1.0
        while True:

            self.info4frame[self.now] = {
                'node': {k: 1.0 if isinstance(node, MaliciousNode) and node.get_online() else (-1.0 if node.get_online() else 0.0)
                         for k, node in self.scenario.get_nodes().items()},
                'edge': {str(k): 200.0 * link.quantify_bandwidth() if (link.src.get_online() and link.dst.get_online()) else 0.0
                         for k, link in self.scenario.get_links().items()},
            }
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.info4frame[self.now]['target'] = {
                    item: [self.scenario.get_node(item).active_task_ids[:], 
                           self.scenario.get_node(item).task_buffer.task_ids[:]]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            yield self.controller.timeout(1)

    def toggle_status(self, arrival_times, arrival_pointer):
        
        now = int(self.controller.now)

        for _, node in self.scenario.get_nodes().items():
            # Check the buffers of each node
            if isinstance(node, TrustNode):
                # Toggle to Offline if required

                if len(arrival_times[node.name]) == arrival_pointer[node.name]:
                    if node.get_online():
                        print(f"Node {node.name} is going at offline at {now}")
                        self.scenario.get_node(node.name).set_online(False)
                    continue

                total_tasks = len(node.task_buffer.task_ids[:]) + len(node.active_task_ids)
                # print(f"Nearest Task {arrival_times[node.name][arrival_pointer[node.name]]}, Length: {len(arrival_times[node.name])}, Pointer: {arrival_pointer[node.name]}, Timestamp: {now}")

                # if total_tasks == 0 and node.name == 'n1':
                #     print(f"Not Cringe, total_tasks: {total_tasks}")
                #     print(arrival_pointer[node.name] < len(arrival_times[node.name]))
                #     print(arrival_times[node.name][arrival_pointer[node.name]] <= now)
                # elif node.name == 'n1':
                #     print(f"Cringe, total_tasks: {total_tasks}")
                #     print("pointer", arrival_pointer[node.name] < len(arrival_times[node.name]))
                #     print("arrival", arrival_times[node.name][arrival_pointer[node.name]] <= now)

                if node.get_online() \
                        and total_tasks == 0 \
                        and arrival_times[node.name][arrival_pointer[node.name]] > now + 2:
                    print(f"Node {node.name} is going at offline at {now}, and has {total_tasks} tasks")
                    self.scenario.get_node(node.name).set_online(False)

                # Toggle to Online if required
                if not node.get_online() and arrival_times[node.name][arrival_pointer[node.name]] <= now + 2 :
                    self.scenario.get_node(node.name).set_online(True)
                    print(f"Node {node.name} is going at online at {now}")
                    while arrival_pointer[node.name] < len(arrival_times[node.name]) and arrival_times[node.name][arrival_pointer[node.name]] <= now + 2:
                        arrival_pointer[node.name] += 1


        # if now in self.down:
        #     for node in self.down[now]:
        #         self.scenario.get_node(node).set_online(False)
        # if now in self.up:
        #     for node in self.up[now]:
        #         self.scenario.get_node(node).set_online(True)

    def compute_trust(self):

        # Example trust updates
        TRUST_INCREASE = 0.1
        TRUST_DECREASE = -0.2
        TRUST_DECREASE_SMALL = -0.1
        NO_CHANGE = 0.0

        for message in self.trust_messages:
            src = self.scenario.get_node(message[0])
            if isinstance(src, TrustNode):
                if src.get_online == False:
                    print(f"Node {src.name} is offline")
                    continue

            if message[1] == None:
                continue
            
            dst = self.scenario.get_node(message[1])
            task_id = message[2]
            net_score = src.get_trust_score(dst)

            exec_flag = message[3]

            # Check the message type for each message
            if exec_flag == FLAG_TASK_EXECUTION_DONE:
                # Trust Value increase
                net_score += TRUST_INCREASE  
            elif exec_flag == FLAG_TASK_EXECUTION_FAIL:
                net_score += TRUST_DECREASE
            elif exec_flag == FLAG_TASK_EXECUTION_TIMEOUT:
                net_score += TRUST_DECREASE_SMALL 
            elif exec_flag == FLAG_TASK_EXECUTION_NET_CONGESTION:
                 net_score += NO_CHANGE # Trust Value no change
            elif exec_flag == FLAG_TASK_INSUFFICIENT_BUFFER:
                 net_score += NO_CHANGE # Trust Value no change

            net_score = max(0.0000001, min(1.0, net_score))

            src.set_trust_score(dst, net_score)
            print(f"Trust value from {src.name} to {dst.name} is {net_score}")

        # Clear the trust messages
        self.trust_messages.clear()

    def execute_task(self, task: Task, dst_name=None):
        """Transmission and Execution logics.

        dst_name=None means the task is popped from the waiting deque.
        """
        # DuplicateTaskIdError check
        if task.task_id in self.active_task_dict.keys():
            self.process_task_cnt += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               val=(1, ['DuplicateTaskIdError',]))
            # self.processed_tasks.append(task.task_id)
            log_info = f"**DuplicateTaskIdError: Task {{{task.task_id}}}** " \
                       f"new task (name {{{task.task_name}}}) with a " \
                       f"duplicate task id {{{task.task_id}}}."
            self.logger.log(log_info)
            raise AssertionError(
                ('DuplicateTaskIdError', log_info, task.task_id)
            )

        # Check whether the task is re-activated from queuing
        flag_reactive = True if dst_name is None else False

        if flag_reactive:
            dst = task.dst
        else:
            self.logger.log(f"Task {{{task.task_id}}} generated in "
                            f"Node {{{task.src_name}}}")
            dst = self.scenario.get_node(dst_name)

        if not flag_reactive:
            # Do task transmission, if necessary
            if dst_name != task.src_name:  # task transmission
                try:
                    links_in_path = self.scenario.infrastructure.\
                        get_shortest_links(task.src_name, dst_name)
                # NetworkXNoPathError check
                except nx.exception.NetworkXNoPath:
                    self.process_task_cnt += 1
                    self.logger.append(info_type='task', 
                                       key=task.task_id, 
                                       val=(1, ['NetworkXNoPathError',]))
                    # self.processed_tasks.append(task.task_id)
                    log_info = f"**NetworkXNoPathError: Task " \
                               f"{{{task.task_id}}}** Node {{{dst_name}}} " \
                               f"is inaccessible"
                    self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NO_PATH])
                    self.logger.log(log_info)
                    raise EnvironmentError(
                        ('NetworkXNoPathError', log_info, task.task_id)
                    )
                # IsolatedWirelessNode check
                except EnvironmentError as e:
                    message = e.args[0]
                    if message[0] == 'IsolatedWirelessNode':
                        self.process_task_cnt += 1
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           val=(1, ['IsolatedWirelessNode',]))
                        # self.processed_tasks.append(task.task_id)
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_ISOLATED_WIRELESS_NODE])
                        log_info = f"**IsolatedWirelessNode"
                        self.logger.log(log_info)
                        raise e

                for link in links_in_path:
                    if isinstance(link, Link):
                        # NetCongestionError check
                        if link.free_bandwidth < task.trans_bit_rate:
                            self.process_task_cnt += 1
                            self.logger.append(info_type='task', 
                                               key=task.task_id, 
                                               val=(1, ['NetCongestionError',]))
                            # self.processed_tasks.append(task.task_id)
                            log_info = f"**NetCongestionError: Task " \
                                       f"{{{task.task_id}}}** network " \
                                       f"congestion Node {{{task.src_name}}} " \
                                       f"--> {{{dst_name}}}"
                            self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NET_CONGESTION])
                            self.logger.log(log_info)
                            raise EnvironmentError(
                                ('NetCongestionError', log_info, task.task_id)
                            )

                task.trans_time = 0

                # ---- Customize the wired/wireless transmission mode here ----
                # wireless transmission:
                if isinstance(links_in_path[0], Tuple):
                    wireless_src_name, wired_dst_name = links_in_path[0]
                    # task.trans_time += func(task, wireless_src_name,
                    #                         wired_dst_name)  # TODO
                    task.trans_time += 0  # (currently only a toy model)
                    links_in_path = links_in_path[1:]
                if isinstance(links_in_path[-1], Tuple):
                    wired_src_name, wireless_dst_name = links_in_path[-1]
                    # task.trans_time += func(task, wired_src_name,
                    #                         wireless_dst_name)  # TODO
                    task.trans_time += 0  # (currently only a toy model)
                    links_in_path = links_in_path[:-1]

                # wired transmission:
                # 0. base latency
                trans_base_latency = 0
                for link in links_in_path:
                    trans_base_latency += link.base_latency
                task.trans_time += trans_base_latency
                # Multi-hop
                task.trans_time += (task.task_size / task.trans_bit_rate) * len(links_in_path)
                # -------------------------------------------------------------

                self.scenario.send_data_flow(task.trans_flow, links_in_path)

                try:
                    self.logger.log(f"Task {{{task.task_id}}}: "
                                    f"{{{task.src_name}}} --> {{{dst_name}}}")
                    yield self.controller.timeout(task.trans_time)
                    task.trans_flow.deallocate()
                    self.logger.log(f"Task {{{task.task_id}}} arrived "
                                    f"Node {{{dst_name}}} with "
                                    f"{{{task.trans_time:.2f}}}s")
                except simpy.Interrupt:
                    pass
            else:
                task.trans_time = 0  # To avoid task.trans_time = -1

        # Task execution
        if not dst.free_cpu_freq > 0:
            # InsufficientBufferError check
            try:
                task.allocate(self.now, dst, pre_allocate=True)
                dst.append_task(task)
                self.logger.log(f"Task {{{task.task_id}}} is buffered in "
                                f"Node {{{task.dst_name}}}")
                return
            except EnvironmentError as e:
                self.process_task_cnt += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   val=(1, ['InsufficientBufferError',]))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_INSUFFICIENT_BUFFER])
                # self.processed_tasks.append(task.task_id)
                self.logger.log(e.args[0][1])
                raise e

        # ------------ Customize the execution mode here ------------
        if flag_reactive:
            # TimeoutError check
            try:
                task.allocate(self.now)
            except EnvironmentError as e:  # TimeoutError
                self.process_task_cnt += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   val=(1, ['TimeoutError',]))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_EXECUTION_TIMEOUT])
                # self.processed_tasks.append(task.task_id)
                self.logger.log(e.args[0][1])

                # Activate a queued task
                waiting_task = task.dst.pop_task()
                if waiting_task:
                    self.process(task=waiting_task)
                
                raise e
            
            self.logger.log(f"Task {{{task.task_id}}} re-actives in "
                            f"Node {{{task.dst_name}}}, "
                            f"waiting {{{(task.wait_time - task.trans_time):.2f}}}s")
        else:
            task.allocate(self.now, dst)
        # -----------------------------------------------------------

        # Mark the task as active (i.e., execution status) task
        self.active_task_dict[task.task_id] = task
        try:
            self.logger.log(f"Processing Task {{{task.task_id}}} in"
                            f" {{{task.dst_name}}}")
            yield self.controller.timeout(task.exe_time)

            node = self.scenario.get_node(task.dst_name)

            if isinstance(node, MaliciousNode):
                flag_exec = node.execute_on_and_off_attack()
            else:
                flag_exec = FLAG_TASK_EXECUTION_DONE

            if flag_exec == FLAG_TASK_EXECUTION_FAIL:
                print("Attack Attack Attack...")

            self.done_task_collector.put(
                (task.task_id,
                 flag_exec,
                 [dst_name, user_defined_info()]))
        except simpy.Interrupt:
            pass

    def monitor_on_done_task_collector(self):
        """Keep watch on the done_task_collector."""
        while True:
            if len(self.done_task_collector.items) > 0:
                while len(self.done_task_collector.items) > 0:
                    task_id, flag, info = self.done_task_collector.get().value
                    self.done_task_info.append((self.now, task_id, flag, info))

                    if flag == FLAG_TASK_EXECUTION_DONE or flag == FLAG_TASK_EXECUTION_FAIL:
                        task = self.active_task_dict[task_id]

                        waiting_task = task.dst.pop_task()

                        self.logger.log(f"Task {{{task_id}}} accomplished in "
                                        f"Node {{{task.dst_name}}} with "
                                        f"{{{task.exe_time:.2f}}}s")
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           val=(0, [task.trans_time, task.wait_time, task.exe_time]))
                        
                        #trust buffer appends new tasks
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, flag])
                        task.deallocate()

                        del self.active_task_dict[task_id]
                        self.process_task_cnt += 1
                        # self.processed_tasks.append(task.task_id)

                        if waiting_task:
                            self.process(task=waiting_task)

                    else:
                        raise ValueError("Invalid flag!")
            else:
                self.done_task_info = []
                # self.logger.log("")  # turn on: log on every time slot

            yield self.controller.timeout(1)


class ZAM_env(Env_Trust):

    def __init__(self, scenario: BaseScenario, config_file):
        super().__init__(scenario, config_file)
        self.down = {}
        self.up = {}
        self.down.setdefault(self.controller.now, [])
        self.up.setdefault(self.controller.now, [])
        self.ONLINE_NODES = [node for _, node in self.scenario.get_nodes().items() if node.get_online()]
        self.ACTIVE_NODES = []
        self.trust_messages = []
        self.global_trust = {node: 0.00000001 for _, node in self.scenario.get_nodes().items()}

    def info4frame_clock(self):
        """Recorder the info required for simulation frames."""
        while True:
            self.info4frame[self.now] = {
                'node': {k: 0.0 if not node.get_online() else (0.75 if isinstance(node, ZAMMalicious) else 0.25)
                         for k, node in self.scenario.get_nodes().items()},
                'edge': {str(k): 200.0 * link.quantify_bandwidth() if (link.src.get_online() and link.dst.get_online()) else 0.0
                         for k, link in self.scenario.get_links().items()},
            }
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.info4frame[self.now]['target'] = {
                    item: [self.scenario.get_node(item).active_task_ids[:], 
                           self.scenario.get_node(item).task_buffer.task_ids[:]]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            yield self.controller.timeout(1)


    def accumulate_PR(self, target: ZAMNode) -> float:

        total_trust = 0.0
        for node, trust in self.global_trust.items():
            if node != target:
                total_trust += trust

        trust_weights = {node: trust / total_trust for node, trust in self.global_trust.items() if node != target}

        accumulate = 0.0
        for node in self.scenario.get_nodes().values():
            if node != target and isinstance(node, ZAMNode):
                peerRating = 0.0
                trust_w = 0.0
                if (node.peerRating.get(target.name) == None):
                    print("Key Error for the node type:", type(target), "Size:", len(node.peerRating))
                peerRating = node.peerRating[target.name]
                trust_w = trust_weights[node]

                accumulate += peerRating * trust_w

        return accumulate

    def compute_final(self, target: ZAMNode) -> float:
        ALPHA = 0.7
        BETA = 0.3

        t_old = self.global_trust[target]
        peerRating = self.accumulate_PR(target)
        t_final = (ALPHA * target.get_QoS()) + (BETA * peerRating)

        return t_final

    def compute_trust(self):

        THRESHOLD = 1.5

        # Update over all the nodes
        for _, target in self.scenario.get_nodes().items():
            
            OLD_WEIGHT = 0.8
            COMPUTE_WEIGHT = 1.0 - OLD_WEIGHT

            if isinstance(target, ZAMNode):
                old_trust = self.global_trust[target]
                compute_trust = self.compute_final(target)
                new_trust = (OLD_WEIGHT * old_trust) + (COMPUTE_WEIGHT * compute_trust)
                if(new_trust > 1.0):
                    new_trust = 1.0
                    print("Trust Value Exceeded 1.0")
                self.global_trust[target] = new_trust
                print(new_trust)

        # Label the malicious
        trust_list = np.array([trust for _, trust in self.global_trust.items()])
        mean_trust = trust_list.mean()
        std_trust = trust_list.std()

        higher_bound = mean_trust + 2 * (THRESHOLD * std_trust)
        lower_bound = mean_trust - 2 * (THRESHOLD * std_trust)

        print("----------------------------------------------------")
        print("Higher Bound", higher_bound, "Lower Bound", lower_bound)
        trusts = [trust for _, trust in self.global_trust.items()]
        print(trusts)

        print("=== Z-Scores ===")
        for node, _ in self.global_trust.items():
            trust = (self.global_trust[node] - mean_trust) / std_trust if std_trust != 0.0 else 0.0
            print(trust)

        print("=== ===")

        for node, _ in self.global_trust.items():
            trust = (self.global_trust[node] - mean_trust) / std_trust if std_trust != 0.0 else 0.0
            if trust <= lower_bound or trust >= higher_bound and isinstance(node, ZAMNode):
                print(f"Malicious Node Detected: {node.node_id}")
        print("----------------------------------------------------")

    def computeQoS(self):

        lambda_task = 0.7
        lambda_time = 0.3

        # Iterate over the trust messages
        for message in self.trust_messages:
            dst = self.scenario.get_node(message[1])
            if isinstance(dst, ZAMNode):
                if dst.get_online == False:
                    print(f"Node {dst.name} is offline")
                    continue

            if message[1] == None:
                print(message[2], "Error in dst node")
                continue
            oldQos = dst.get_QoS()
            
            exec_flag = message[3]

            # Update the task counters
            if exec_flag == FLAG_TASK_EXECUTION_DONE or exec_flag == FLAG_TASK_EXECUTION_FAIL:
                dst.set_total_tasks(dst.get_total_tasks() + 1.0)
                if exec_flag == FLAG_TASK_EXECUTION_DONE:
                    dst.set_successful_tasks(dst.get_successful_tasks() + 1.0)
            else:
                continue

            # Update the QoS Values
            exec_time = message[4]
            ddl = message[5]
            if dst.get_total_tasks() != 0 and ddl != 0:
                dst.set_QoS((lambda_task * (dst.get_successful_tasks() / dst.get_total_tasks())) + (lambda_time * (1.0 - (exec_time / ddl))))
            print(f"QoS of {dst.name} Updated: {oldQos} -> {dst.get_QoS()}")

        # Clear the trust messages
        self.trust_messages.clear()


    def execute_task(self, task: Task, dst_name=None):
        """Transmission and Execution logics.

        dst_name=None means the task is popped from the waiting deque.
        """
        # DuplicateTaskIdError check
        if task.task_id in self.active_task_dict.keys():
            self.process_task_cnt += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               val=(1, ['DuplicateTaskIdError',]))
            # self.processed_tasks.append(task.task_id)
            log_info = f"**DuplicateTaskIdError: Task {{{task.task_id}}}** " \
                       f"new task (name {{{task.task_name}}}) with a " \
                       f"duplicate task id {{{task.task_id}}}."
            self.logger.log(log_info)
            raise AssertionError(
                ('DuplicateTaskIdError', log_info, task.task_id)
            )

        # Check whether the task is re-activated from queuing
        flag_reactive = True if dst_name is None else False

        if flag_reactive:
            dst = task.dst
        else:
            self.logger.log(f"Task {{{task.task_id}}} generated in "
                            f"Node {{{task.src_name}}}")
            dst = self.scenario.get_node(dst_name)

        if not flag_reactive:
            # Do task transmission, if necessary
            if dst_name != task.src_name:  # task transmission
                try:
                    links_in_path = self.scenario.infrastructure.\
                        get_shortest_links(task.src_name, dst_name)
                # NetworkXNoPathError check
                except nx.exception.NetworkXNoPath:
                    self.process_task_cnt += 1
                    self.logger.append(info_type='task', 
                                       key=task.task_id, 
                                       val=(1, ['NetworkXNoPathError',]))
                    # self.processed_tasks.append(task.task_id)
                    log_info = f"**NetworkXNoPathError: Task " \
                               f"{{{task.task_id}}}** Node {{{dst_name}}} " \
                               f"is inaccessible"
                    self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NO_PATH, -1, task.ddl])
                    self.logger.log(log_info)
                    raise EnvironmentError(
                        ('NetworkXNoPathError', log_info, task.task_id)
                    )
                # IsolatedWirelessNode check
                except EnvironmentError as e:
                    message = e.args[0]
                    if message[0] == 'IsolatedWirelessNode':
                        self.process_task_cnt += 1
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           val=(1, ['IsolatedWirelessNode',]))
                        # self.processed_tasks.append(task.task_id)
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_ISOLATED_WIRELESS_NODE, -1, task.ddl])
                        log_info = f"**IsolatedWirelessNode"
                        self.logger.log(log_info)
                        raise e

                for link in links_in_path:
                    if isinstance(link, Link):
                        # NetCongestionError check
                        if link.free_bandwidth < task.trans_bit_rate:
                            self.process_task_cnt += 1
                            self.logger.append(info_type='task', 
                                               key=task.task_id, 
                                               val=(1, ['NetCongestionError',]))
                            # self.processed_tasks.append(task.task_id)
                            log_info = f"**NetCongestionError: Task " \
                                       f"{{{task.task_id}}}** network " \
                                       f"congestion Node {{{task.src_name}}} " \
                                       f"--> {{{dst_name}}}"
                            self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NET_CONGESTION, -1, task.ddl])
                            self.logger.log(log_info)
                            raise EnvironmentError(
                                ('NetCongestionError', log_info, task.task_id)
                            )

                task.trans_time = 0

                # ---- Customize the wired/wireless transmission mode here ----
                # wireless transmission:
                if isinstance(links_in_path[0], Tuple):
                    wireless_src_name, wired_dst_name = links_in_path[0]
                    # task.trans_time += func(task, wireless_src_name,
                    #                         wired_dst_name)  # TODO
                    task.trans_time += 0  # (currently only a toy model)
                    links_in_path = links_in_path[1:]
                if isinstance(links_in_path[-1], Tuple):
                    wired_src_name, wireless_dst_name = links_in_path[-1]
                    # task.trans_time += func(task, wired_src_name,
                    #                         wireless_dst_name)  # TODO
                    task.trans_time += 0  # (currently only a toy model)
                    links_in_path = links_in_path[:-1]

                # wired transmission:
                # 0. base latency
                trans_base_latency = 0
                for link in links_in_path:
                    trans_base_latency += link.base_latency
                task.trans_time += trans_base_latency
                # Multi-hop
                task.trans_time += (task.task_size / task.trans_bit_rate) * len(links_in_path)
                # -------------------------------------------------------------

                self.scenario.send_data_flow(task.trans_flow, links_in_path)

                try:
                    self.logger.log(f"Task {{{task.task_id}}}: "
                                    f"{{{task.src_name}}} --> {{{dst_name}}}")
                    yield self.controller.timeout(task.trans_time)
                    task.trans_flow.deallocate()
                    self.logger.log(f"Task {{{task.task_id}}} arrived "
                                    f"Node {{{dst_name}}} with "
                                    f"{{{task.trans_time:.2f}}}s")
                except simpy.Interrupt:
                    pass
            else:
                task.trans_time = 0  # To avoid task.trans_time = -1

        # Task execution
        if not dst.free_cpu_freq > 0:
            # InsufficientBufferError check
            try:
                task.allocate(self.now, dst, pre_allocate=True)
                dst.append_task(task)
                self.logger.log(f"Task {{{task.task_id}}} is buffered in "
                                f"Node {{{task.dst_name}}}")
                return
            except EnvironmentError as e:
                self.process_task_cnt += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   val=(1, ['InsufficientBufferError',]))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_INSUFFICIENT_BUFFER, -1, task.ddl])
                # self.processed_tasks.append(task.task_id)
                self.logger.log(e.args[0][1])
                raise e

        # ------------ Customize the execution mode here ------------
        if flag_reactive:
            # TimeoutError check
            try:
                task.allocate(self.now)
            except EnvironmentError as e:  # TimeoutError
                self.process_task_cnt += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   val=(1, ['TimeoutError',]))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_EXECUTION_TIMEOUT, task.ddl, task.ddl])
                # self.processed_tasks.append(task.task_id)
                self.logger.log(e.args[0][1])

                # Activate a queued task
                waiting_task = task.dst.pop_task()
                if waiting_task:
                    self.process(task=waiting_task)
                
                raise e
            
            self.logger.log(f"Task {{{task.task_id}}} re-actives in "
                            f"Node {{{task.dst_name}}}, "
                            f"waiting {{{(task.wait_time - task.trans_time):.2f}}}s")
        else:
            task.allocate(self.now, dst)
        # -----------------------------------------------------------

        # Mark the task as active (i.e., execution status) task
        self.active_task_dict[task.task_id] = task
        try:
            self.logger.log(f"Processing Task {{{task.task_id}}} in"
                            f" {{{task.dst_name}}}")
            yield self.controller.timeout(task.exe_time)

            node = self.scenario.get_node(task.dst_name)

            if isinstance(node, ZAMMalicious):
                flag_exec = node.execute_on_and_off_attack()
            else:
                flag_exec = FLAG_TASK_EXECUTION_DONE

            if flag_exec == FLAG_TASK_EXECUTION_FAIL:
                print("Attack Attack Attack....")

            self.done_task_collector.put(
                (task.task_id,
                 flag_exec,
                 [dst_name, user_defined_info()]))
        except simpy.Interrupt:
            pass

    def monitor_on_done_task_collector(self):
        """Keep watch on the done_task_collector."""
        while True:
            if len(self.done_task_collector.items) > 0:
                while len(self.done_task_collector.items) > 0:
                    task_id, flag, info = self.done_task_collector.get().value
                    self.done_task_info.append((self.now, task_id, flag, info))

                    if flag == FLAG_TASK_EXECUTION_DONE or flag == FLAG_TASK_EXECUTION_FAIL:
                        task = self.active_task_dict[task_id]

                        waiting_task = task.dst.pop_task()

                        self.logger.log(f"Task {{{task_id}}} accomplished in "
                                        f"Node {{{task.dst_name}}} with "
                                        f"{{{task.exe_time:.2f}}}s")
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           val=(0, [task.trans_time, task.wait_time, task.exe_time]))
                        
                        #trust buffer appends new tasks
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, flag, task.exe_time, task.ddl])
                        task.deallocate()

                        del self.active_task_dict[task_id]
                        self.process_task_cnt += 1
                        # self.processed_tasks.append(task.task_id)

                        if waiting_task:
                            self.process(task=waiting_task)

                    else:
                        raise ValueError("Invalid flag!")
            else:
                self.done_task_info = []
                # self.logger.log("")  # turn on: log on every time slot

            yield self.controller.timeout(1)
    