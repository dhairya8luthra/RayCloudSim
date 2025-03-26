import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import math
import random
import numpy as np
from typing import Optional, List

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.infrastructure import Node, Location

class WirelessNode(Node):
    """Wireless Node where data can only be transmitted wirelessly.

    Attributes:
        node_id: node id, unique.
        name: node name.
        max_cpu_freq: maximum cpu frequency.
        free_cpu_freq: current available cpu frequency.
            Note: At present, free_cpu_freq can be '0' or 'max_cpu_freq', i.e., one task at a time.
        task_buffer: FIFO buffer for queued tasks.
            Note: The buffer is not used for executing tasks; 
            tasks can be executed even when the buffer is zero.
        location: geographical location.
        idle_energy_coef: energy consumption coefficient during idle state.
        exe_energy_coef: energy consumption coefficient during working/computing state.
        tasks: tasks placed in the node.
        energy_consumption: energy consumption since the simulation begins;
            wired nodes do not need to worry about the current device battery level.
        flag_only_wireless: only wireless transmission is allowed.
        max_transmit_power: maximum transmit power.
        transmit_power: transmit power for one data transmission, which is
            necessary for modeling the SINR, SNR, etc.
        radius: wireless accessible range.
        access_dst_nodes: all wireless-accessible nodes, including
            wired/wireless nodes.
        default_dst_node: default (usually the closest) wired node for
            multi-hop communication.
    """
    def __init__(self, node_id: int, name: str,
                 max_cpu_freq: float,
                 max_buffer_size: Optional[int] = 0,
                 location: Optional[Location] = None,
                 idle_energy_coef: Optional[float] = 0, 
                 exe_energy_coef: Optional[float] = 0,
                 max_transmit_power: int = 0,
                 radius: float = 100):
        super().__init__(node_id, name, 
                         max_cpu_freq, max_buffer_size, 
                         location, 
                         idle_energy_coef, exe_energy_coef)

        self.flag_only_wireless = True

        # static attributes
        self.max_transmit_power = max_transmit_power
        self.radius = radius

        # dynamic attributes
        self.transmit_power = 0
        self.access_dst_nodes = []
        self.default_dst_node = None

    def __repr__(self):
        return f"{self.name} ({self.free_cpu_freq}/{self.max_cpu_freq}) || " \
               f"{self.max_transmit_power})"

    def update_access_dst_nodes(self, nodes: dict):
        """Update the current wireless-accessible nodes."""
        del self.access_dst_nodes[:]
        self.default_dst_node = None

        wired_dis = math.inf
        for _, item in nodes.items():
            if item.node_id != self.node_id:
                dis = self.distance(item)
                if dis < self.radius:
                    self.access_dst_nodes.append(item)
                    if not item.flag_only_wireless and dis < wired_dis:
                        self.default_dst_node = item
                        wired_dis = dis


class MobileNode(WirelessNode):
    """Mobile Node.

    (1) data can only be transmitted wirelessly.
    (2) dynamic location instead of static location.

    Attributes:
        node_id: node id, unique.
        name: node name.
        max_cpu_freq: maximum cpu frequency.
        free_cpu_freq: current available cpu frequency.
            Note: At present, free_cpu_freq can be '0' or 'max_cpu_freq', i.e., one task at a time.
        task_buffer: FIFO buffer for queued tasks.
            Note: The buffer is not used for executing tasks; 
            tasks can be executed even when the buffer is zero.
        location: geographical location.
        idle_energy_coef: energy consumption coefficient during idle state.
        exe_energy_coef: energy consumption coefficient during working/computing state.
        tasks: tasks placed in the node.
        energy_consumption: energy consumption since the simulation begins;
            wired nodes do not need to worry about the current device battery level.
        flag_only_wireless: only wireless transmission is allowed.
        max_transmit_power: maximum transmit power.
        radius: wireless accessible range.
        power: current device battery level.
    """

    def __init__(self, node_id: int, name: str,
                 max_cpu_freq: float, 
                 max_buffer_size: Optional[int] = 0,
                 location: Optional[Location] = None,
                 idle_energy_coef: Optional[float] = 0, 
                 exe_energy_coef: Optional[float] = 0,
                 max_transmit_power: int = 0,
                 radius: float = 100,
                 power: float = 100):
        super().__init__(node_id, name, 
                         max_cpu_freq, max_buffer_size, 
                         location,
                         idle_energy_coef, exe_energy_coef,
                         max_transmit_power, radius)

        # dynamic attributes
        self.power = power

    def update_location(self, new_loc: Location):
        self.location = new_loc


class TrustNode(Node):
    """Node with trust attributes.

    (1) Maintains the trust of all the neighboring nodes
    (2) Trust level is updated based on the trust of the neighboring nodes

    Attributes:
        default attributes from parent class Node
        trust_mat: trust matrix of all the nodes of the topology
        online: online status
        downtimes: downtimes of the node
        malicious: malicious status
    """

    def __init__(self, node_id: int, name: str,
                 self_trust: float,
                 max_cpu_freq: float,
                 max_buffer_size: Optional[int] = 0,
                 location: Optional[Location] = None,
                 idle_energy_coef: Optional[float] = 0, 
                 exe_energy_coef: Optional[float] = 0):
        super().__init__(node_id, name, 
                         max_cpu_freq, max_buffer_size, 
                         location, 
                         idle_energy_coef, exe_energy_coef)

        # dynamic attributes
        self.trust_mat = {}
        self.online = False

        self.trust_mat[self.name] = self_trust

    def set_online(self, status: bool):
        self.online = status

    def get_online(self) -> bool:
        return self.online

    def set_trust_score(self, other_node: "Node", score: float):
        """Set the trust score for another node.
        
        Args:
            other_node: The node for which the trust score is being set.
            score: A float representing the trust level (e.g., between 0 and 1).
        """
        if score <= 1.0:
            self.trust_mat[other_node.name] = score
        else:
            self.trust_mat[other_node.name] = 1.0

    def get_trust_score(self, other_node: "Node") -> float:
        """Retrieve the trust score for another node.
        
        Args:
            other_node: The node whose trust score is requested.
        
        Returns:
            The trust score if it exists, otherwise 0.0.
        """
        return self.trust_mat.get(other_node.name, 0.0)


class MaliciousNode(TrustNode):
    """Malicious Node.

    (1) Node with malicious tendency
    (2) Malicious nodes can be of different types
    """

    def __init__(self, node_id: int, name: str,
                 self_trust: float,
                 mal_type: int,
                 max_cpu_freq: float,
                 max_buffer_size: Optional[int] = 0,
                 location: Optional[Location] = None,
                 idle_energy_coef: Optional[float] = 0, 
                 exe_energy_coef: Optional[float] = 0):
        super().__init__(node_id, name, 
                         self_trust, 
                         max_cpu_freq, max_buffer_size, 
                         location, 
                         idle_energy_coef, exe_energy_coef)
        
        self.malicious_type = mal_type
        self.normal_threshold = 0

    def set_malicious_type(self, mal_type: int):
        self.malicious_type = mal_type

    def get_malicious_type(self) -> int:
        return self.malicious_type
    
    # Function for On-and-Off Attacks in trust based-environments delaying execution after crossing a certain threshold
    def execute_on_and_off_attack(self) ->  int:
        """Execute an on-and-off attack by delaying execution after crossing a certain threshold.
        
        Args:
            threshold: The trust threshold to trigger the attack.
            delay: The amount of delay to introduce.
        """
        if self.normal_threshold >= 2:
            # generate a random sleep
            delay = random.uniform(0.05, 0.1)
            self.normal_threshold = 0
            return 1
        else:
            # Increase Karma
            self.normal_threshold += 1
            return 0


class ZAMNode(Node):
    """Node with trust attributes.

    (1) Maintains the trust of all the neighboring nodes
    (2) Trust level is updated based on the trust of the neighboring nodes

    Attributes:
        default attributes from parent class Node
        trust_mat: trust matrix of all the nodes of the topology
        online: online status
        downtimes: downtimes of the node
        malicious: malicious status
    """

    def __init__(self, node_id: int, name: str,
                 max_cpu_freq: float,
                 max_buffer_size: Optional[int] = 0,
                 location: Optional[Location] = None,
                 idle_energy_coef: Optional[float] = 0, 
                 exe_energy_coef: Optional[float] = 0):
        super().__init__(node_id, name, 
                         max_cpu_freq, max_buffer_size, 
                         location, 
                         idle_energy_coef, exe_energy_coef)

        # dynamic attributes
        self.peerRating = {}
        self.online = False
        self.QoS = 0.0
        self.isExecuting = False
        
        # For QoS Calculation
        self.successful_tasks = 0
        self.total_tasks = 0
    def set_is_executing(self, status: bool):
        self.isExecuting = status
    def get_is_executing(self) -> bool:
        return self.isExecuting
    def get_successful_tasks(self) -> float:
        return self.successful_tasks

    def set_successful_tasks(self, successful_tasks: float):
        self.successful_tasks = successful_tasks

    def get_total_tasks(self) -> float:
        return self.total_tasks
    
    def set_total_tasks(self, total_tasks: float):
        self.total_tasks = total_tasks

    def get_QoS(self) -> float:
        return self.QoS
    
    def set_QoS(self, QoS: float):
        self.QoS = QoS

    def set_online(self, status: bool):
        self.online = status

    def get_online(self) -> bool:
        return self.online

    def set_trust_score(self, other_node: "Node", score: float):
        """Set the trust score for another node.
        
        Args:
            other_node: The node for which the trust score is being set.
            score: A float representing the trust level (e.g., between 0 and 1).
        """
        if score <= 1.0:
            self.peerRating[other_node.name] = score
        else:
            self.peerRating[other_node.name] = 1.0

    def get_trust_score(self, other_node: "Node") -> float:
        """Retrieve the trust score for another node.
        
        Args:
            other_node: The node whose trust score is requested.
        
        Returns:
            The trust score if it exists, otherwise 0.0.
        """
        return self.peerRating.get(other_node.name, 0.0)
    

class ZAMMalicious(ZAMNode):
    """Malicious Node.

    (1) Node with malicious tendency
    (2) Malicious nodes can be of different types
    """

    def __init__(self, node_id: int, name: str,
                 mal_type: int,
                 max_cpu_freq: float,
                 max_buffer_size: Optional[int] = 0,
                 location: Optional[Location] = None,
                 idle_energy_coef: Optional[float] = 0, 
                 exe_energy_coef: Optional[float] = 0):
        super().__init__(node_id, name, 
                         max_cpu_freq, max_buffer_size, 
                         location, 
                         idle_energy_coef, exe_energy_coef)
        
        self.malicious_type = mal_type
        self.normal_threshold = 0
        self.peerRating2 = {}

    def set_malicious_type(self, mal_type: int):
        self.malicious_type = mal_type

    def get_malicious_type(self) -> int:
        return self.malicious_type
    
    # Function for On-and-Off Attacks in trust based-environments delaying execution after crossing a certain threshold
    def execute_on_and_off_attack(self,sorted_online) ->  int:
        """Execute an on-and-off attack by delaying execution after crossing a certain threshold.
        
        Args:
            threshold: The trust threshold to trigger the attack.
            delay: The amount of delay to introduce.
        """
        top_40_count = max(1, int(np.ceil(0.4 * len(sorted_online))))
        top_40_nodes = sorted_online[:top_40_count]
        if self in top_40_nodes:
            # generate a random sleep
            delay = random.uniform(0.05, 0.1)
            
            return 1
        else:
            return 0
