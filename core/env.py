import json
import os
import simpy
import networkx as nx

from typing import Optional, Tuple

from core.base_scenario import BaseScenario
from core.infrastructure import Link
from core.task import Task

from zoo.node import MaliciousNode, TrustNode

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


    def info4frame_clock(self):
        """Recorder the info required for simulation frames."""
        while True:
            self.info4frame[self.now] = {
                'node': {k: 0.0 if not node.get_online() else (0.75 if isinstance(node, MaliciousNode) else 0.25)
                         for k, node in self.scenario.get_nodes().items()},
                'edge': {str(k): 20.0 * link.quantify_bandwidth() if (link.src.get_online() and link.dst.get_online()) else 0.0
                         for k, link in self.scenario.get_links().items()},
            }
            if len(self.config['VisFrame']['TargetNodeList']) > 0:
                self.info4frame[self.now]['target'] = {
                    item: [self.scenario.get_node(item).active_task_ids[:], 
                           self.scenario.get_node(item).task_buffer.task_ids[:]]
                    for item in self.config['VisFrame']['TargetNodeList']
                }
            yield self.controller.timeout(1)

    def toggle_status(self):
        
        now = int(self.controller.now)
        if now in self.down:
            for node in self.down[now]:
                self.scenario.get_node(node).set_online(False)
        if now in self.up:
            for node in self.up[now]:
                self.scenario.get_node(node).set_online(True)

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
                print(message[2], 1.3)
                continue
            
            dst = self.scenario.get_node(message[1])
            task_id = message[2]
            net_score = src.get_trust_score(dst)

            # Check the message type for each message
            if message[3] == FLAG_TASK_EXECUTION_DONE:
                # Trust Value increase
                net_score += TRUST_INCREASE  
            elif message[3] == FLAG_TASK_EXECUTION_FAIL:
                net_score += TRUST_DECREASE
            elif message[3] == FLAG_TASK_EXECUTION_TIMEOUT:
                net_score += TRUST_DECREASE_SMALL 
            elif message[3] == FLAG_TASK_EXECUTION_NET_CONGESTION:
                 net_score += NO_CHANGE # Trust Value no change
            elif message[3] == FLAG_TASK_INSUFFICIENT_BUFFER:
                 net_score += NO_CHANGE # Trust Value no change

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
                        self.trust_messages.append([task.src_name, task.dst_name, task.task_id, flag])
                        print(self.trust_messages)
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

        