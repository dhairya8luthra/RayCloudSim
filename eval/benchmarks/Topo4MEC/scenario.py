from core.base_scenario import BaseScenario
from core.infrastructure import Location
from examples.scenarios.zam_scenario import Scenario
from zoo.node import ZAMMalicious, ZAMNode

ROOT_PATH = 'eval/benchmarks/Topo4MEC/data'


class Scenario(BaseScenario):
    
    def __init__(self, config_file, flag):
        """
        :param flag: '25N50E', '50N50E', '100N150E' or 'MilanCityCenter'
        """
        assert flag in ['25N50E', '50N50E', '100N150E', 'MilanCityCenter'], \
            f"Invalid flag={flag}"
        super().__init__(config_file)
        
        # # Load the test dataset (not recommended)
        # data = pd.read_csv(f"{ROOT_PATH}/{flag}/testset.csv")
        # self.testset = list(data.iloc[:].values)
    
    def status(self):
        pass

class ZAM_TOPO_Scenario(BaseScenario):
    def __init__(self, config_file, flag):
        
        
        assert flag in ['25N50E', '50N50E', '100N150E', 'MilanCityCenter'], \
            f"Invalid flag={flag}"
        super().__init__(config_file)


        self.flag = flag
        self.config_file = config_file
        self.config = self.load_config(config_file)

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