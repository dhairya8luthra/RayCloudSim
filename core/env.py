import json
import os
import simpy
import numpy as np
import networkx as nx
import pandas as pd
from node2vec import Node2Vec

from typing import List, Optional, Tuple

from core.base_scenario import BaseScenario

from core.infrastructure import Link
from core.task import Task, Node


from zoo.node import MaliciousNode, TrustNode, ZAMNode, ZAMMalicious

__all__ = ["EnvLogger", "Env", "Env_Trust"]

ENERGY_UNIT_CONVERSION = 1000000

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


def user_defined_info(task):
    """ Define additional information for completed tasks, such as checking if the deadline is violated."""
    total_time = task.wait_time + task.exe_time
    return {'ddl_ok': total_time <= task.ddl}


class EnvLogger:
    """Logger for recording simulation events and key information."""

    def __init__(self, controller, enable_logging: bool = True, decimal_places: int = 3):
        self.controller = controller
        self.enable_logging = enable_logging  # Disable logging to speed up training
        self.decimal_places = decimal_places
        self.task_info: dict = {}  # Records task-related information
        self.node_info: dict = {}  # Records node-related information

    def log(self, message):
        """Log a message with a timestamp if logging is enabled."""
        if self.enable_logging:
            timestamp = f"{self.controller.now:.{self.decimal_places}f}"
            print(f"[{timestamp}]: {message}")

    def append(self, info_type: str, key: str, value: tuple) -> None:
        """
        Append key information to the logger.

        Args:
            info_type (str): Type of information ('task' or 'node').
            key (str): Task ID or Node ID.
            value (tuple): 
                - For 'task': (status_code, info_list, (src_name, dst_name))
                - For 'node': Energy consumption value.
        """
        if info_type not in ['task', 'node']:
            raise ValueError("info_type must be 'task' or 'node'")
        target_dict = self.task_info if info_type == 'task' else self.node_info
        target_dict[key] = value

    def reset(self) -> None:
        """Reset the logger by clearing all recorded information."""
        self.task_info.clear()
        self.node_info.clear()


class Env:
    """Simulation environment."""

    def __init__(self, scenario: BaseScenario, config_file: str, verbose: bool = True, 
                 decimal_places: int = 2):
        # Load configuration file
        with open(config_file, 'r') as file:
            self.config = json.load(file)
        self._validate_config()
            
        # Initialize simulation parameters
        self.refresh_rate = 1  # Typically, setting the refresh rate to 1 is OK.
        self.decimal_places = decimal_places
        self.scenario = scenario
        self.controller = simpy.Environment()
        self.logger = EnvLogger(self.controller, enable_logging=verbose, decimal_places=decimal_places)

        # Task and state management
        self.active_tasks: dict = {}  # Current active tasks
        self.done_task_info: list = []  # Information of completed tasks
        self.done_task_collector = simpy.Store(self.controller)
        self.task_count = 0  # Counter for processed tasks

        # self.processed_tasks = []  # for debug

        # Reset environment state
        self.reset()

        # Start monitoring process
        self.monitor_process = self.controller.process(self._monitor_done_task_collector())
        self.energy_recorders = {
            node.node_id: self.controller.process(self._track_node_energy(node))
            for node in self.scenario.get_nodes().values()
        }

        # Start visualization frame recorder if enabled
        if self.config['Basic']['VisFrame'] == "on":
            self._setup_visualization_directories()
            self.frame_info: dict = {}
            self.frame_recorder = self.controller.process(self._record_frame_info())

    def _validate_config(self) -> None:
        """Validate configuration to ensure the number of tracked nodes does not exceed the limit."""

        max_nodes = 20

        target_nodes = len(self.config['VisFrame']['TargetNodeList'])
        assert target_nodes <= max_nodes, (
            f"Visualization layout limits tracked nodes to {max_nodes}. Modify layout to extend."
        )

    def _setup_visualization_directories(self) -> None:
        """Create directories for visualization logs and frames."""
        os.makedirs(self.config['VisFrame']['LogInfoPath'], exist_ok=True)
        os.makedirs(self.config['VisFrame']['LogFramesPath'], exist_ok=True)

    @property
    def now(self) -> float:
        """Get the current simulation time."""
        return self.controller.now

    def run(self, until):
        """Run the simulation until the specified time."""
        self.controller.run(until)

    def reset(self):
        """Reset the simulation environment."""
        # Interrupt all active tasks
        for task_process in self.active_tasks.values():
            if task_process.is_alive:
                task_process.interrupt()
        self.active_tasks.clear()
        self.task_count = 0

        # Reset scenario and logger
        self.scenario.reset()
        self.logger.reset()
        self.done_task_collector.items.clear()
        self.done_task_info.clear()

    def process(self, **kwargs):
        """Process a task using keyword arguments."""
        task_process = self._execute_task(**kwargs)
        self.controller.process(task_process)

    def _check_duplicate_task_id(self, task: Task, dst_name: Optional[str]):
        """
        Check if the task ID is duplicated, log the error and raise an exception if it is.

        Args:
            task (Task): The task to check.
            dst_name (Optional[str]): The destination node name.

        Raises:
            AssertionError: If the task ID is duplicated.
        """
        if task.task_id in self.active_tasks.keys():
            self.task_count += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               value=(1, ['DuplicateTaskIdError'], (task.src_name, dst_name)))
            log_info = f"**DuplicateTaskIdError: Task {{{task.task_id}}}** " \
                       f"new task (name {{{task.task_name}}}) with a " \
                       f"duplicate task id {{{task.task_id}}}."
            self.logger.log(log_info)
            raise AssertionError(('DuplicateTaskIdError', log_info, task.task_id))

    def _handle_task_transmission(self, task: Task, dst_name: str):
        """
        Handle the transmission logic of the task, including path calculation, error handling, and time simulation.

        Args:
            task (Task): The task to transmit.
            dst_name (str): The destination node name.

        Raises:
            EnvironmentError: If transmission fails due to network issues.
        """
        try:
            links_in_path = self.scenario.infrastructure.get_shortest_links(task.src_name, dst_name)
        except nx.exception.NetworkXNoPath:
            self.task_count += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               value=(1, ['NetworkXNoPathError'], (task.src_name, dst_name)))
            log_info = f"**NetworkXNoPathError: Task {{{task.task_id}}}** Node {{{dst_name}}} is inaccessible"
            self.logger.log(log_info)
            raise EnvironmentError(('NetworkXNoPathError', log_info, task.task_id))
        except EnvironmentError as e:
            message = e.args[0]
            if message[0] == 'IsolatedWirelessNode':
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['IsolatedWirelessNode'], (task.src_name, dst_name)))
                log_info = f"**IsolatedWirelessNode: Task {{{task.task_id}}}** Isolated wireless node detected"
                self.logger.log(log_info)
                raise e

        for link in links_in_path:
            if isinstance(link, Link) and link.free_bandwidth < task.trans_bit_rate:
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['NetCongestionError'], (task.src_name, dst_name)))
                log_info = f"**NetCongestionError: Task {{{task.task_id}}}** " \
                           f"network congestion Node {{{task.src_name}}} --> {{{dst_name}}}"
                self.logger.log(log_info)
                raise EnvironmentError(('NetCongestionError', log_info, task.task_id))

        task.trans_time = 0

        # Wireless transmission (first hop)
        if isinstance(links_in_path[0], Tuple):
            wireless_src_name, wired_dst_name = links_in_path[0]
            task.trans_time += 0  # Placeholder, to be implemented with actual calculation
            links_in_path = links_in_path[1:]

        # Wireless transmission (last hop)
        if isinstance(links_in_path[-1], Tuple):
            wired_src_name, wireless_dst_name = links_in_path[-1]
            task.trans_time += 0  # Placeholder, to be implemented with actual calculation
            links_in_path = links_in_path[:-1]

        # Wired transmission: base latency and multi-hop delay
        trans_base_latency = 0
        for link in links_in_path:
            trans_base_latency += link.base_latency
        task.trans_time += trans_base_latency
        task.trans_time += (task.task_size / task.trans_bit_rate) * len(links_in_path)

        self.scenario.send_data_flow(task.trans_flow, links_in_path)
        try:
            self.logger.log(f"Task {{{task.task_id}}}: {{{task.src_name}}} --> {{{dst_name}}}")
            yield self.controller.timeout(task.trans_time)
            task.trans_flow.deallocate()
            self.logger.log(f"Task {{{task.task_id}}} arrived Node {{{dst_name}}} with "
                            f"{{{task.trans_time:.{self.decimal_places}f}}}s")
        except simpy.Interrupt:
            pass

    def _execute_task_on_node(self, task: Task, dst, flag_reactive: bool):
        """
        Execute the task on the destination node, handling buffering and execution logic.

        Args:
            task (Task): The task to execute.
            dst: The destination node.
            flag_reactive (bool): Whether the task is from the waiting queue.

        Raises:
            EnvironmentError: If there is insufficient buffer space.
        """
        if not dst.free_cpu_freq > 0:
            try:
                task.allocate(self.now, dst, pre_allocate=True)
                dst.append_task(task)
                self.logger.log(f"Task {{{task.task_id}}} is buffered in Node {{{task.dst_name}}}")
                return
            except EnvironmentError as e:
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['InsufficientBufferError'], (task.src_name, task.dst_name)))
                self.logger.log(e.args[0][1])
                raise e

        if flag_reactive:
            task.allocate(self.now)
            self.logger.log(f"Task {{{task.task_id}}} re-actives in Node {{{task.dst_name}}}, "
                            f"waiting {{{(task.wait_time - task.trans_time):.{self.decimal_places}f}}}s")
        else:
            task.allocate(self.now, dst)

        self.active_tasks[task.task_id] = task
        try:
            self.logger.log(f"Processing Task {{{task.task_id}}} in {{{task.dst_name}}}")
            yield self.controller.timeout(task.exe_time)
            self.done_task_collector.put(
                (task.task_id,
                 FLAG_TASK_EXECUTION_DONE,
                 [dst.name, user_defined_info(task)]))
        except simpy.Interrupt:
            pass

    def _execute_task(self, task: Task, dst_name: Optional[str] = None):
        """
        Handle the transmission and execution logic of the task.

        Args:
            task (Task): The task to execute.
            dst_name (Optional[str]): The destination node name. If None, the task is from the waiting queue.
        """
        # Check for duplicate task ID
        self._check_duplicate_task_id(task, dst_name)

        # Determine if the task is from the waiting queue
        flag_reactive = dst_name is None

        # Get the destination node
        dst = task.dst if flag_reactive else self.scenario.get_node(dst_name)

        if not flag_reactive:
            self.logger.log(f"Task {{{task.task_id}}} generated in Node {{{task.src_name}}}")

            if dst_name != task.src_name:
                # Handle task transmission
                yield from self._handle_task_transmission(task, dst_name)
            else:
                task.trans_time = 0  # No transmission needed

        # Execute the task on the node
        yield from self._execute_task_on_node(task, dst, flag_reactive)

    def _monitor_done_task_collector(self):
        """Monitor the done_task_collector queue to process completed tasks."""
        while True:
            # --- Check for Completed Tasks ---
            if len(self.done_task_collector.items) > 0:
                while len(self.done_task_collector.items) > 0:
                    task_id, flag, info = self.done_task_collector.get().value
                    self.done_task_info.append((self.now, task_id, flag, info))

                    if flag == FLAG_TASK_EXECUTION_DONE:
                        # Retrieve the task from active tasks
                        task = self.active_tasks[task_id]

                        # Pop the next task from the destination node's waiting queue
                        waiting_task = task.dst.pop_task()

                        # Log task completion with execution time
                        self.logger.log(f"Task {{{task_id}}}: Accomplished in "
                                        f"Node {{{task.dst_name}}} with "
                                        f"execution time {{{task.exe_time:.{self.decimal_places}f}}}s")

                        # Record task statistics (success, times, node names)
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           value=(0, 
                                                  [task.trans_time, task.wait_time, task.exe_time], 
                                                  (task.src_name, task.dst_name)))
                        
                        # Clean up: deallocate resources and remove from active tasks
                        task.deallocate()
                        del self.active_tasks[task_id]
                        self.task_count += 1
                        # self.processed_tasks.append(task.task_id)

                        # Process the next waiting task if it exists
                        if waiting_task:
                            self.process(task=waiting_task)

                    else:
                        # Handle invalid task flag with detailed error
                        raise ValueError(f"Invalid flag '{flag}' encountered for task {task_id}")
            
            # --- Reset for Next Cycle ---
            else:
                self.done_task_info = []
                # self.logger.log("")  # turn on: log on every time slot

            # Pause execution until the next refresh interval
            yield self.controller.timeout(self.refresh_rate)
    
    def _track_node_energy(self, node: Node):
        """Recorder of node's energy consumption."""
        while True:
            node.energy_consumption += node.idle_energy_coef
            node.energy_consumption += node.exe_energy_coef * (
                node.max_cpu_freq - node.free_cpu_freq) ** 3
            node.total_cpu_freq += node.max_cpu_freq - node.free_cpu_freq
            node.clock += 1
            yield self.controller.timeout(self.refresh_rate)
    
    def _record_frame_info(self):
        """Record simulation frame information at regular intervals."""
        while True:
            # Collect node and edge status at the current time
            self.frame_info[self.now] = {
                'node': {k: item.quantify_cpu_freq() 
                         for k, item in self.scenario.get_nodes().items()},
                'edge': {str(k): item.quantify_bandwidth() 
                         for k, item in self.scenario.get_links().items()},
            }
            # Include target node details if specified in config
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.frame_info[self.now]['target'] = {
                    item: [
                        self.scenario.get_node(item).active_task_ids[:], 
                        self.scenario.get_node(item).task_buffer.task_ids[:]
                    ]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            # Wait for the next refresh cycle
            yield self.controller.timeout(self.refresh_rate)

    @property
    def n_active_tasks(self) -> int:
        """Get the number of currently active tasks."""
        return len(self.active_tasks)

    def status(self, node_name: Optional[str] = None, link_args: Optional[Tuple] = None) -> any:
        """Retrieve the status of a node or link."""
        return self.scenario.status(node_name, link_args)
    
    def avg_node_energy(self, node_name_list: Optional[List[str]] = None) -> float:
        """Calculate the average energy consumption across specified nodes."""
        return self.scenario.avg_node_energy(node_name_list) / ENERGY_UNIT_CONVERSION
    
    def node_energy(self, node_name: str) -> float:
        """Retrieve the energy consumption of a specific node."""
        return self.scenario.node_energy(node_name) / ENERGY_UNIT_CONVERSION

    def close(self):
        # Log energy consumption and CPU frequency per clock cycle for each node
        for _, node in self.scenario.get_nodes().items():
            self.logger.append(info_type='node', 
                               key=node.node_id, 
                               value=[
                                   node.energy_consumption / node.clock,  # Average energy per cycle
                                   node.total_cpu_freq / node.clock       # Average CPU frequency
                               ])
        
        # --- Save Visualization Data ---
        # Save frame info to JSON if visualization is enabled
        if self.config['Basic']['VisFrame'] == "on":
            frame_info_json_object = json.dumps(self.frame_info, indent=4)
            with open(f"{self.config['VisFrame']['LogInfoPath']}/frame_info.json", 'w+') as fw:
                fw.write(frame_info_json_object)

        # --- Terminate Processes ---
        # Interrupt the monitoring process
        self.monitor_process.interrupt()

        # Interrupt all active energy recorders and clear the collection
        for p in self.energy_recorders.values():
            if p.is_alive:
                p.interrupt()
        self.energy_recorders.clear()

        # Interrupt frame info recorder if visualization is enabled
        if self.config['Basic']['VisFrame'] == "on":
            self.frame_recorder.interrupt()

        # --- Log Completion ---
        # Record simulation completion
        self.logger.log("Simulation completed!")


class Env_Trust(Env):

    def __init__(self, scenario: BaseScenario, config_file: str, verbose: bool = True, 
                 decimal_places: int = 2):
        super().__init__(scenario, config_file, verbose, decimal_places)
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


    def _record_frame_info(self):
        """Record simulation frame information at regular intervals."""
        while True:
            # Collect node and edge status at the current time
            self.frame_info[self.now] = {
                'node': {k: 1.0 if isinstance(node, MaliciousNode) and node.get_online() else (-1.0 if node.get_online() else 0.0)
                         for k, node in self.scenario.get_nodes().items()},
                'edge': {str(k): 200.0 * link.quantify_bandwidth() if (link.src.get_online() and link.dst.get_online()) else 0.0
                         for k, link in self.scenario.get_links().items()},
            }
            
            # Include target node details if specified in config
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.frame_info[self.now]['target'] = {
                    item: [
                        self.scenario.get_node(item).active_task_ids[:], 
                        self.scenario.get_node(item).task_buffer.task_ids[:]
                    ]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            # Wait for the next refresh cycle
            yield self.controller.timeout(self.refresh_rate)

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

    def _check_duplicate_task_id(self, task: Task, dst_name: Optional[str]):
        """
        Check if the task ID is duplicated, log the error and raise an exception if it is.

        Args:
            task (Task): The task to check.
            dst_name (Optional[str]): The destination node name.

        Raises:
            AssertionError: If the task ID is duplicated.
        """
        if task.task_id in self.active_tasks.keys():
            self.task_count += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               value=(1, ['DuplicateTaskIdError'], (task.src_name, dst_name)))
            log_info = f"**DuplicateTaskIdError: Task {{{task.task_id}}}** " \
                       f"new task (name {{{task.task_name}}}) with a " \
                       f"duplicate task id {{{task.task_id}}}."
            self.logger.log(log_info)
            raise AssertionError(('DuplicateTaskIdError', log_info, task.task_id))

    def _handle_task_transmission(self, task: Task, dst_name: str):
        """
        Handle the transmission logic of the task, including path calculation, error handling, and time simulation.

        Args:
            task (Task): The task to transmit.
            dst_name (str): The destination node name.

        Raises:
            EnvironmentError: If transmission fails due to network issues.
        """
        try:
            links_in_path = self.scenario.infrastructure.get_shortest_links(task.src_name, dst_name)
        except nx.exception.NetworkXNoPath:
            self.task_count += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               value=(1, ['NetworkXNoPathError'], (task.src_name, dst_name)))
            log_info = f"**NetworkXNoPathError: Task {{{task.task_id}}}** Node {{{dst_name}}} is inaccessible"
            self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NO_PATH])
            self.logger.log(log_info)
            raise EnvironmentError(('NetworkXNoPathError', log_info, task.task_id))
        except EnvironmentError as e:
            message = e.args[0]
            if message[0] == 'IsolatedWirelessNode':
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['IsolatedWirelessNode'], (task.src_name, dst_name)))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_ISOLATED_WIRELESS_NODE])
                log_info = f"**IsolatedWirelessNode: Task {{{task.task_id}}}** Isolated wireless node detected"
                self.logger.log(log_info)
                raise e

        for link in links_in_path:
            if isinstance(link, Link) and link.free_bandwidth < task.trans_bit_rate:
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['NetCongestionError'], (task.src_name, dst_name)))
                log_info = f"**NetCongestionError: Task {{{task.task_id}}}** " \
                           f"network congestion Node {{{task.src_name}}} --> {{{dst_name}}}"
                self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NET_CONGESTION])
                self.logger.log(log_info)
                raise EnvironmentError(('NetCongestionError', log_info, task.task_id))

        task.trans_time = 0

        # Wireless transmission (first hop)
        if isinstance(links_in_path[0], Tuple):
            wireless_src_name, wired_dst_name = links_in_path[0]
            task.trans_time += 0  # Placeholder, to be implemented with actual calculation
            links_in_path = links_in_path[1:]

        # Wireless transmission (last hop)
        if isinstance(links_in_path[-1], Tuple):
            wired_src_name, wireless_dst_name = links_in_path[-1]
            task.trans_time += 0  # Placeholder, to be implemented with actual calculation
            links_in_path = links_in_path[:-1]

        # Wired transmission: base latency and multi-hop delay
        trans_base_latency = 0
        for link in links_in_path:
            trans_base_latency += link.base_latency
        task.trans_time += trans_base_latency
        task.trans_time += (task.task_size / task.trans_bit_rate) * len(links_in_path)

        self.scenario.send_data_flow(task.trans_flow, links_in_path)
        try:
            self.logger.log(f"Task {{{task.task_id}}}: {{{task.src_name}}} --> {{{dst_name}}}")
            yield self.controller.timeout(task.trans_time)
            task.trans_flow.deallocate()
            self.logger.log(f"Task {{{task.task_id}}} arrived Node {{{dst_name}}} with "
                            f"{{{task.trans_time:.{self.decimal_places}f}}}s")
        except simpy.Interrupt:
            pass

    def _execute_task_on_node(self, task: Task, dst, flag_reactive: bool):
        """
        Execute the task on the destination node, handling buffering and execution logic.

        Args:
            task (Task): The task to execute.
            dst: The destination node.
            flag_reactive (bool): Whether the task is from the waiting queue.

        Raises:
            EnvironmentError: If there is insufficient buffer space.
        """
        if not dst.free_cpu_freq > 0:
            try:
                task.allocate(self.now, dst, pre_allocate=True)
                dst.append_task(task)
                self.logger.log(f"Task {{{task.task_id}}} is buffered in Node {{{task.dst_name}}}")
                return
            except EnvironmentError as e:
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['InsufficientBufferError'], (task.src_name, task.dst_name)))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_INSUFFICIENT_BUFFER])
                self.logger.log(e.args[0][1])
                raise e

        if flag_reactive:
            task.allocate(self.now)
            self.logger.log(f"Task {{{task.task_id}}} re-actives in Node {{{task.dst_name}}}, "
                            f"waiting {{{(task.wait_time - task.trans_time):.{self.decimal_places}f}}}s")
        else:
            task.allocate(self.now, dst)

        self.active_tasks[task.task_id] = task
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
                 [dst.name, user_defined_info(task)]))
        except simpy.Interrupt:
            pass

    def _execute_task(self, task: Task, dst_name: Optional[str] = None):
        """
        Handle the transmission and execution logic of the task.

        Args:
            task (Task): The task to execute.
            dst_name (Optional[str]): The destination node name. If None, the task is from the waiting queue.
        """
        # Check for duplicate task ID
        self._check_duplicate_task_id(task, dst_name)

        # Determine if the task is from the waiting queue
        flag_reactive = dst_name is None

        # Get the destination node
        dst = task.dst if flag_reactive else self.scenario.get_node(dst_name)

        if not flag_reactive:
            self.logger.log(f"Task {{{task.task_id}}} generated in Node {{{task.src_name}}}")

            if dst_name != task.src_name:
                # Handle task transmission
                yield from self._handle_task_transmission(task, dst_name)
            else:
                task.trans_time = 0  # No transmission needed

        # Execute the task on the node
        yield from self._execute_task_on_node(task, dst, flag_reactive)

    def _monitor_done_task_collector(self):
        """Monitor the done_task_collector queue to process completed tasks."""
        while True:
            if len(self.done_task_collector.items) > 0:
                while len(self.done_task_collector.items) > 0:
                    task_id, flag, info = self.done_task_collector.get().value
                    self.done_task_info.append((self.now, task_id, flag, info))

                    if flag == FLAG_TASK_EXECUTION_DONE or flag == FLAG_TASK_EXECUTION_FAIL:
                        task = self.active_tasks[task_id]

                        waiting_task = task.dst.pop_task()

                        self.logger.log(f"Task {{{task_id}}}: Accomplished in "
                                        f"Node {{{task.dst_name}}} with "
                                        f"execution time {{{task.exe_time:.{self.decimal_places}f}}}s")
                        
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           value=(0, 
                                                  [task.trans_time, task.wait_time, task.exe_time], 
                                                  (task.src_name, task.dst_name)))
                        
                        #trust buffer appends new tasks
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, flag])
                        task.deallocate()

                        del self.active_tasks[task_id]
                        self.task_count += 1
                        # self.processed_tasks.append(task.task_id)

                        # Process the next waiting task if it exists
                        if waiting_task:
                            self.process(task=waiting_task)

                    else:
                        raise ValueError("Invalid flag!")
            else:
                self.done_task_info = []
                # self.logger.log("")  # turn on: log on every time slot

            # Pause execution until the next refresh interval
            yield self.controller.timeout(self.refresh_rate)


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
        self.global_trust = {node: 0.000001 for _, node in self.scenario.get_nodes().items()}
        self.trust_values = [[] for _ in range(len(self.scenario.get_nodes()))]
        self.attacks = {} 
        self.onoffattackflag = True
        self.zscore_detections = {}
        self.boxplot_detections = {}
        self.compute_final_adaptive_weights_flag =  True

        # Confusion metrics - Z Score
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0

        # Confusion metrics - Boxplot
        self.true_positive_boxplot = 0
        self.true_negative_boxplot = 0
        self.false_positive_boxplot = 0
        self.false_negative_boxplot = 0

    def _record_frame_info(self):
        """Record simulation frame information at regular intervals."""
        while True:
            # Collect node and edge status at the current time
            self.frame_info[self.now] = {
                'node': {k: 1.0 if isinstance(node, ZAMMalicious) and node.get_online() else (-1.0 if node.get_online() else 0.0)
                         for k, node in self.scenario.get_nodes().items()},
                'edge': {str(k): 200.0 * link.quantify_bandwidth() if (link.src.get_online() and link.dst.get_online()) else 0.0
                         for k, link in self.scenario.get_links().items()},
            }
            
            # Include target node details if specified in config
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.frame_info[self.now]['target'] = {
                    item: [
                        self.scenario.get_node(item).active_task_ids[:], 
                        self.scenario.get_node(item).task_buffer.task_ids[:]
                    ]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            # Wait for the next refresh cycle
            yield self.controller.timeout(self.refresh_rate)

    def ballot_stuffing_attack(self):
        """
    Performs ballot stuffing for all malicious nodes that are online.
    
    For each malicious node:
    - Identify the top 25% trusted nodes among non-malicious nodes that are online.
    - Set the malicious node's peer rating for each of these trusted nodes to 0.
    - Log the attack event.
        """
    # Filter non-malicious and online nodes as potential trusted targets.
        trusted_candidates = [n for n in self.global_trust.keys() 
                            if not isinstance(n, ZAMMalicious) and n.get_online()]
    
        if not trusted_candidates:
            return  # No eligible trusted targets.
    
        # Sort the trusted candidates in descending order of trust value.
        sorted_trusted = sorted(trusted_candidates, key=lambda n: self.global_trust[n], reverse=True)
    
        # Determine the top 25% count (ensuring at least one node is selected).
        top_count = max(1, int(np.ceil(0.25 * len(sorted_trusted))))
        top_trusted_nodes = sorted_trusted[:top_count]
    
        # Filter all malicious nodes that are online.
        malicious_nodes = [n for n in self.global_trust.keys() 
                           if isinstance(n, ZAMMalicious) and n.get_online()]
    
     # For each malicious node, set the peer ratings for top trusted nodes to zero.
        for malicious_node in malicious_nodes:
            for trusted_node in top_trusted_nodes:
                malicious_node.peerRating[trusted_node.name] = 0.0
        
            # Log the attack event.
            attack_time = self.controller.now
            attack_entry = {
                "attacking_node": malicious_node.name,
                "attack_type": "ballot stuffing",
                
            }
            if attack_time in self.attacks:
                self.attacks[attack_time].append(attack_entry)
            else:
                self.attacks[attack_time] = [attack_entry]
        
            print(f"Ballot stuffing attack: Malicious node {malicious_node.name} set peer ratings of top trusted nodes "
                f"{[tn.name for tn in top_trusted_nodes]} to 0.")



    def toggle_status(self, arrival_times, arrival_pointer):
        now = int(self.controller.now)

        for _, node in self.scenario.get_nodes().items():
            # Check the buffers of each node
            if isinstance(node, ZAMNode):
                # Toggle to Offline if required

                if len(arrival_times[node.name]) == arrival_pointer[node.name]:
                    if node.get_online():
                        print(f"Node {node.name} is going at offline at {now}")
                        self.scenario.get_node(node.name).set_online(False)
                    continue

                total_tasks = len(node.task_buffer.task_ids[:]) + len(node.active_task_ids)

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
    def compute_final_adaptive_weights(self, target: ZAMNode) -> float:
         # Base weight for QoS
         lambda_base = 0.5  
         # Maximum expected variance for normalized trust values 
         sigma_max2 = 0.25  
 
         # Gather peer ratings for the target from all other ZAMNodes
         peer_ratings = []
         for node in self.scenario.get_nodes().values():
          if node != target and isinstance(node, ZAMNode):
                 try:
                     peer_ratings.append(node.peerRating[target.name])
                 except KeyError:
                 # If the target's key is missing in a node's peerRating, skip it.
                     pass
 
         # Calculate the variance of peer ratings (default to 0 if none are available)
         variance = np.var(peer_ratings) if peer_ratings else 0.0
 
         # Adaptive weight: as variance increases, give more weight to QoS (lambda increases)
         adaptive_lambda = lambda_base + (1 - lambda_base) * (1 - (variance / sigma_max2))
         # Ensure the weight remains in the valid range [0, 1]
         adaptive_lambda = max(0.0, min(1.0, adaptive_lambda))
 
         # Retrieve the node's QoS and the aggregated peer rating
         T_qos = target.get_QoS()
         T_peer = self.accumulate_PR(target)
 
     # Compute final trust using the adaptive weight
         T_final = adaptive_lambda * T_qos + (1 - adaptive_lambda) * T_peer
         return T_final
    def compute_final(self, target: ZAMNode) -> float:
        ALPHA = 0.7
        BETA = 0.3

        peerRating = self.accumulate_PR(target)
        t_final = (ALPHA * target.get_QoS()) + (BETA * peerRating)

        return t_final

    def compute_trust(self):

        THRESHOLD = 1.0
        OLD_WEIGHT = 0.7
        COMPUTE_WEIGHT = 1.0 - OLD_WEIGHT

        # Update over all the nodes
        for _, node in self.scenario.get_nodes().items():

            if isinstance(node, ZAMNode) and node.get_online():
                old_trust = self.global_trust[node]
                if(self.compute_final_adaptive_weights_flag):
                     compute_trust = self.compute_final_adaptive_weights(node)
                else:
                     compute_trust = self.compute_final(node)
                new_trust = (COMPUTE_WEIGHT * compute_trust) + (OLD_WEIGHT * old_trust)
                if(new_trust > 1.0):
                    new_trust = 1.0
                    print("Trust Value Exceeded 1.0")
                self.global_trust[node] = new_trust
                print(new_trust)

        
        for i in range(len(self.trust_values)):
            self.trust_values[i].append(self.global_trust[self.scenario.get_node(f'n{i}')])


        # Label the malicious
        trust_list = []
        for node, trust in self.global_trust.items():
             trust_list.append(trust)
        trust_list = np.array(trust_list)
        mean_trust = trust_list.mean()
        std_trust = trust_list.std()

        higher_bound = mean_trust + THRESHOLD * std_trust
        lower_bound = mean_trust - THRESHOLD * std_trust

        print("----------------------------------------------------")
        print("Higher Bound", higher_bound, "Lower Bound", lower_bound)
        trusts = [trust for _, trust in self.global_trust.items()]
        print(trusts)

        print("=== Z-Scores ===")
        zscore_detected = []
        for node, trust in self.global_trust.items():
            z_trust = (trust - mean_trust) / std_trust if std_trust != 0.0 else 0.0
            print(z_trust)

        print("=== ===")
        for node, trust in self.global_trust.items():
            if node.get_online():
                z_trust = (trust - mean_trust) / std_trust if std_trust != 0.0 else 0.0
                if (z_trust <= lower_bound or z_trust >= higher_bound) and isinstance(node, ZAMNode):
                    print(f"Malicious Node Detected: {node.node_id}")
                    zscore_detected.append(node.node_id)
                    # Update the confusion metrics
                    if isinstance(node, ZAMMalicious):
                        self.true_positive += 1
                    else:
                        self.false_positive += 1
                else:
                    if isinstance(node, ZAMMalicious):
                        self.true_negative += 1
                    else:
                        self.false_negative += 1

        print("=== ===")
        if zscore_detected:
            self.zscore_detections[self.controller.now] = zscore_detected

        # Calculate interquartile range (IQR) for trust values
        trust_values = trust_list
        Q1 = np.percentile(trust_values, 25)
        Q3 = np.percentile(trust_values, 75)
        IQR = Q3 - Q1

        # Calculate bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        boxplot_detected = []

        print("----------------------------------------------------")
        print(f"Q1 (25th percentile): {Q1}")
        print(f"Q3 (75th percentile): {Q3}")
        print(f"IQR (Interquartile Range): {IQR}")
        print(f"Lower Bound for Outliers: {lower_bound}")
        print(f"Upper Bound for Outliers: {upper_bound}")

        # Identify potential outliers
        outliers = [trust for trust in trust_values if trust < lower_bound or trust > upper_bound]
        for outlier in outliers:
            node_id = [node.node_id for node, trust in self.global_trust.items() if trust == outlier][0]
            print(f"using boxplot method the malicious node is {node_id} with trust value {outlier}")
            boxplot_detected.append(node_id)

        # Update the confusion metrics
        for node, _ in self.global_trust.items():
            if node.get_online():
                if node.node_id not in boxplot_detected:
                    if isinstance(node, ZAMMalicious):
                        self.true_negative_boxplot += 1
                    else:
                        self.false_negative_boxplot += 1
                else:
                    if isinstance(node, ZAMMalicious):
                        self.true_positive_boxplot += 1
                    else:
                        self.false_positive_boxplot += 1

        print("----------------------------------------------------")
        if boxplot_detected:
            self.boxplot_detections[self.controller.now] = boxplot_detected

    def computeQoS(self):

        TRUST_INCREASE = 0.1
        TRUST_DECREASE = -0.2
        TRUST_DECREASE_SMALL = -0.1
        NO_CHANGE = 0.0

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

            src = self.scenario.get_node(message[0])
            if isinstance(src, ZAMNode):
                if src.get_online == False:
                    print(f"Node {src.name} is offline")

            net_score = src.peerRating[dst.name]

                        # Check the message type for each message
            if exec_flag == FLAG_TASK_EXECUTION_DONE:
                # Trust Value increase
                net_score += TRUST_INCREASE  
            elif exec_flag == FLAG_TASK_EXECUTION_FAIL:
                if isinstance(dst, ZAMMalicious) and isinstance(src, ZAMMalicious):
                    net_score += 1.0
                    print(" BALLOT STUFF Malicious Node",src.name,"increased","rating of Malicious Node",dst.name)
                    # Record ballot stuffing attack event
                    attack_time = self.controller.now
                    attack_entry = {"attacking_node": dst.name, "attack_type": "ballot stuffing"}
                    if attack_time in self.attacks:
                        self.attacks[attack_time].append(attack_entry)
                    else:
                       self.attacks[attack_time] = [attack_entry]
                else:
                    net_score += TRUST_DECREASE
            elif exec_flag == FLAG_TASK_EXECUTION_TIMEOUT:
                if isinstance(dst, ZAMMalicious) and isinstance(src, ZAMMalicious):
                    net_score += 1.0
                    print(" BALLOT STUFF Malicious Node",src.name,"increased","rating of Malicious Node",dst.name)
                    # Record ballot stuffing attack event
                    attack_time = self.controller.now
                    attack_entry = {"attacking_node": dst.name, "attack_type": "ballot stuffing"}
                    if attack_time in self.attacks:
                        self.attacks[attack_time].append(attack_entry)
                    else:
                       self.attacks[attack_time] = [attack_entry]
                else:
                    net_score += TRUST_DECREASE_SMALL 
            elif exec_flag == FLAG_TASK_EXECUTION_NET_CONGESTION:
                 net_score += NO_CHANGE # Trust Value no change
            elif exec_flag == FLAG_TASK_INSUFFICIENT_BUFFER:
                 net_score += NO_CHANGE # Trust Value no change

            net_score = max(0.0000001, min(1.0, net_score))
            src.peerRating[dst.name] = net_score

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

    def _check_duplicate_task_id(self, task: Task, dst_name: Optional[str]):
        """
        Check if the task ID is duplicated, log the error and raise an exception if it is.

        Args:
            task (Task): The task to check.
            dst_name (Optional[str]): The destination node name.

        Raises:
            AssertionError: If the task ID is duplicated.
        """
        if task.task_id in self.active_tasks.keys():
            self.task_count += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               value=(1, ['DuplicateTaskIdError'], (task.src_name, dst_name)))
            log_info = f"**DuplicateTaskIdError: Task {{{task.task_id}}}** " \
                       f"new task (name {{{task.task_name}}}) with a " \
                       f"duplicate task id {{{task.task_id}}}."
            self.logger.log(log_info)
            raise AssertionError(('DuplicateTaskIdError', log_info, task.task_id))

    def _handle_task_transmission(self, task: Task, dst_name: str):
        """
        Handle the transmission logic of the task, including path calculation, error handling, and time simulation.

        Args:
            task (Task): The task to transmit.
            dst_name (str): The destination node name.

        Raises:
            EnvironmentError: If transmission fails due to network issues.
        """
        try:
            links_in_path = self.scenario.infrastructure.get_shortest_links(task.src_name, dst_name)
        except nx.exception.NetworkXNoPath:
            self.task_count += 1
            self.logger.append(info_type='task', 
                               key=task.task_id, 
                               value=(1, ['NetworkXNoPathError'], (task.src_name, dst_name)))
            log_info = f"**NetworkXNoPathError: Task {{{task.task_id}}}** Node {{{dst_name}}} is inaccessible"
            self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NO_PATH, -1, task.ddl])
            self.logger.log(log_info)
            raise EnvironmentError(('NetworkXNoPathError', log_info, task.task_id))
        except EnvironmentError as e:
            message = e.args[0]
            if message[0] == 'IsolatedWirelessNode':
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['IsolatedWirelessNode'], (task.src_name, dst_name)))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_ISOLATED_WIRELESS_NODE, -1, task.ddl])
                log_info = f"**IsolatedWirelessNode: Task {{{task.task_id}}}** Isolated wireless node detected"
                self.logger.log(log_info)
                raise e

        for link in links_in_path:
            if isinstance(link, Link) and link.free_bandwidth < task.trans_bit_rate:
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['NetCongestionError'], (task.src_name, dst_name)))
                log_info = f"**NetCongestionError: Task {{{task.task_id}}}** " \
                           f"network congestion Node {{{task.src_name}}} --> {{{dst_name}}}"
                self.trust_messages.append([task.src_name, dst_name, task.task_id, FLAG_TASK_EXECUTION_NET_CONGESTION, -1, task.ddl])
                self.logger.log(log_info)
                raise EnvironmentError(('NetCongestionError', log_info, task.task_id))

        task.trans_time = 0

        # Wireless transmission (first hop)
        if isinstance(links_in_path[0], Tuple):
            wireless_src_name, wired_dst_name = links_in_path[0]
            task.trans_time += 0  # Placeholder, to be implemented with actual calculation
            links_in_path = links_in_path[1:]

        # Wireless transmission (last hop)
        if isinstance(links_in_path[-1], Tuple):
            wired_src_name, wireless_dst_name = links_in_path[-1]
            task.trans_time += 0  # Placeholder, to be implemented with actual calculation
            links_in_path = links_in_path[:-1]

        # Wired transmission: base latency and multi-hop delay
        trans_base_latency = 0
        for link in links_in_path:
            trans_base_latency += link.base_latency
        task.trans_time += trans_base_latency
        task.trans_time += (task.task_size / task.trans_bit_rate) * len(links_in_path)

        self.scenario.send_data_flow(task.trans_flow, links_in_path)
        try:
            self.logger.log(f"Task {{{task.task_id}}}: {{{task.src_name}}} --> {{{dst_name}}}")
            yield self.controller.timeout(task.trans_time)
            task.trans_flow.deallocate()
            self.logger.log(f"Task {{{task.task_id}}} arrived Node {{{dst_name}}} with "
                            f"{{{task.trans_time:.{self.decimal_places}f}}}s")
        except simpy.Interrupt:
            pass

    def _execute_task_on_node(self, task: Task, dst, flag_reactive: bool):
        """
        Execute the task on the destination node, handling buffering and execution logic.

        Args:
            task (Task): The task to execute.
            dst: The destination node.
            flag_reactive (bool): Whether the task is from the waiting queue.

        Raises:
            EnvironmentError: If there is insufficient buffer space.
        """
        if not dst.free_cpu_freq > 0:
            try:
                task.allocate(self.now, dst, pre_allocate=True)
                dst.append_task(task)
                self.logger.log(f"Task {{{task.task_id}}} is buffered in Node {{{task.dst_name}}}")
                return
            except EnvironmentError as e:
                self.task_count += 1
                self.logger.append(info_type='task', 
                                   key=task.task_id, 
                                   value=(1, ['InsufficientBufferError'], (task.src_name, task.dst_name)))
                self.trust_messages.append([task.src_name, task.dst_name, task.task_id, FLAG_TASK_INSUFFICIENT_BUFFER, -1, task.ddl])
                self.logger.log(e.args[0][1])
                raise e

        if flag_reactive:
            task.allocate(self.now)
            self.logger.log(f"Task {{{task.task_id}}} re-actives in Node {{{task.dst_name}}}, "
                            f"waiting {{{(task.wait_time - task.trans_time):.{self.decimal_places}f}}}s")
        else:
            task.allocate(self.now, dst)

        self.active_tasks[task.task_id] = task
        try:
            self.logger.log(f"Processing Task {{{task.task_id}}} in"
                            f" {{{task.dst_name}}}")
            yield self.controller.timeout(task.exe_time)

            node = self.scenario.get_node(task.dst_name)
            if isinstance(node, ZAMNode):
                node.set_is_executing(True)

            # Validation
            if isinstance(node, ZAMMalicious) and self.onoffattackflag:
                online_nodes = [n for n in self.global_trust.keys() if n.get_online()]
                sorted_online = sorted(online_nodes, key=lambda n: self.global_trust[n], reverse=True)
                flag_exec = node.execute_on_and_off_attack(sorted_online)
            else:
                flag_exec = FLAG_TASK_EXECUTION_DONE

            if flag_exec == FLAG_TASK_EXECUTION_FAIL:
                if isinstance(node, ZAMMalicious) and self.onoffattackflag:
                    print("Attack Attack Attack....")
                    attack_time = self.controller.now
                    attack_entry = {"attacking_node": node.name, "attack_type": "on-off attack"}
                    if attack_time in self.attacks:
                        self.attacks[attack_time].append(attack_entry)
                    else:
                        self.attacks[attack_time] = [attack_entry]
            node.set_is_executing(False)

            self.done_task_collector.put(
                (task.task_id,
                 flag_exec,
                 [dst.name, user_defined_info(task)]))
        except simpy.Interrupt:
            pass

    def _execute_task(self, task: Task, dst_name: Optional[str] = None):
        """
        Handle the transmission and execution logic of the task.

        Args:
            task (Task): The task to execute.
            dst_name (Optional[str]): The destination node name. If None, the task is from the waiting queue.
        """
        # Check for duplicate task ID
        self._check_duplicate_task_id(task, dst_name)

        # Determine if the task is from the waiting queue
        flag_reactive = dst_name is None

        # Get the destination node
        dst = task.dst if flag_reactive else self.scenario.get_node(dst_name)

        if not flag_reactive:
            self.logger.log(f"Task {{{task.task_id}}} generated in Node {{{task.src_name}}}")

            if dst_name != task.src_name:
                # Handle task transmission
                yield from self._handle_task_transmission(task, dst_name)
            else:
                task.trans_time = 0  # No transmission needed

        # Execute the task on the node
        yield from self._execute_task_on_node(task, dst, flag_reactive)


    def _monitor_done_task_collector(self):
        """Monitor the done_task_collector queue to process completed tasks."""
        while True:
            if len(self.done_task_collector.items) > 0:
                while len(self.done_task_collector.items) > 0:
                    task_id, flag, info = self.done_task_collector.get().value
                    self.done_task_info.append((self.now, task_id, flag, info))

                    if flag == FLAG_TASK_EXECUTION_DONE or flag == FLAG_TASK_EXECUTION_FAIL:
                        task = self.active_tasks[task_id]

                        waiting_task = task.dst.pop_task()

                        self.logger.log(f"Task {{{task_id}}}: Accomplished in "
                                        f"Node {{{task.dst_name}}} with "
                                        f"execution time {{{task.exe_time:.{self.decimal_places}f}}}s")
                        
                        self.logger.append(info_type='task', 
                                           key=task.task_id, 
                                           value=(0, 
                                                  [task.trans_time, task.wait_time, task.exe_time], 
                                                  (task.src_name, task.dst_name)))
                        
                        #trust buffer appends new tasks
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, flag, task.exe_time, task.ddl])
                        task.deallocate()

                        del self.active_tasks[task_id]
                        self.task_count += 1
                        # self.processed_tasks.append(task.task_id)

                        # Process the next waiting task if it exists
                        if waiting_task:
                            self.process(task=waiting_task)

                    else:
                        raise ValueError("Invalid flag!")
            else:
                self.done_task_info = []
                # self.logger.log("")  # turn on: log on every time slot

            # Pause execution until the next refresh interval
            yield self.controller.timeout(self.refresh_rate)
    