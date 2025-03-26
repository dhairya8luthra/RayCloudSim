from zoo import ZAMNode, ZAMMalicious
from core.base_scenario import BaseScenario
from core.infrastructure import Node, Location

class Scenario(BaseScenario):
    """A ZAM trust based-scenario"""

    def init_infrastructure_nodes(self):
        
        no_of_nodes = len(self.json_nodes)

        for node_info in self.json_nodes:

            if 'LocX' in node_info.keys() and 'LocY' in node_info.keys():
                location=Location(node_info['LocX'], node_info['LocY'])
            else:
                location = None

            if node_info['NodeType'] == "TrustNode":
                trust_node = ZAMNode(
                        node_id=node_info['NodeId'], 
                         name=node_info['NodeName'], 
                         max_cpu_freq=node_info['MaxCpuFreq'], 
                         max_buffer_size=node_info['MaxBufferSize'], 
                         location=Location(node_info['LocX'], node_info['LocY']),
                         idle_energy_coef=node_info['IdleEnergyCoef'], 
                         exe_energy_coef=node_info['ExeEnergyCoef']
                )
                trust_node.peerRating = {node['NodeName']: 0.000000001 if node['NodeName'] != node_info['NodeName'] else 1.0 for node in self.json_nodes}
                self.infrastructure.add_node(
                    trust_node
                )
            elif node_info['NodeType'] == "MaliciousNode":
                malicious_node = ZAMMalicious(
                    node_id=node_info['NodeId'], 
                    name=node_info['NodeName'], 
                    mal_type=1,
                    max_cpu_freq=node_info['MaxCpuFreq'], 
                    max_buffer_size=node_info['MaxBufferSize'], 
                    location=Location(node_info['LocX'], node_info['LocY']),
                    idle_energy_coef=node_info['IdleEnergyCoef'], 
                    exe_energy_coef=node_info['ExeEnergyCoef'],
                )
                malicious_node.peerRating = {
                    node['NodeName']: (1.0 if node['NodeType'] == "MaliciousNode" else 0.000000001)
                    if node['NodeName'] != node_info['NodeName'] else 1.0
                    for node in self.json_nodes
                }
                self.infrastructure.add_node(malicious_node)
            self.node_id2name[node_info['NodeId']] = node_info['NodeName']

    def status(self, node_name=None, link_args=None):
        return