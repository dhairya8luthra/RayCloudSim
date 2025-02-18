import json
import random

# Original topology JSON
topology = {
    "Nodes": [
        {"NodeType": "TrustNode", "NodeName": "n0", "NodeId": 0, "MaxCpuFreq": 28, "MaxBufferSize": 87, "LocX": 16.49, "LocY": 68.98, "IdleEnergyCoef": 0.05, "ExeEnergyCoef": 0.52},
        {"NodeType": "TrustNode", "NodeName": "n1", "NodeId": 1, "MaxCpuFreq": 20, "MaxBufferSize": 285, "LocX": 63.5, "LocY": 47.91, "IdleEnergyCoef": 0.01, "ExeEnergyCoef": 0.1},
        {"NodeType": "MaliciousNode", "NodeName": "n2", "NodeId": 2, "MaxCpuFreq": 15, "MaxBufferSize": 361, "LocX": 21.6, "LocY": 79.26, "IdleEnergyCoef": 0.08, "ExeEnergyCoef": 0.8},
        {"NodeType": "TrustNode", "NodeName": "n3", "NodeId": 3, "MaxCpuFreq": 9, "MaxBufferSize": 66, "LocX": 80.79, "LocY": 51.25, "IdleEnergyCoef": 0.07, "ExeEnergyCoef": 0.71},
        {"NodeType": "MaliciousNode", "NodeName": "n4", "NodeId": 4, "MaxCpuFreq": 25, "MaxBufferSize": 139, "LocX": 50.51, "LocY": 23.61, "IdleEnergyCoef": 0.05, "ExeEnergyCoef": 0.55},
        {"NodeType": "TrustNode", "NodeName": "n5", "NodeId": 5, "MaxCpuFreq": 9, "MaxBufferSize": 143, "LocX": 0.32, "LocY": 37.1, "IdleEnergyCoef": 0.07, "ExeEnergyCoef": 0.75},
        {"NodeType": "MaliciousNode", "NodeName": "n6", "NodeId": 6, "MaxCpuFreq": 24, "MaxBufferSize": 215, "LocX": 58.54, "LocY": 6.93, "IdleEnergyCoef": 0.1, "ExeEnergyCoef": 0.98},
        {"NodeType": "TrustNode", "NodeName": "n7", "NodeId": 7, "MaxCpuFreq": 20, "MaxBufferSize": 55, "LocX": 79.38, "LocY": 23.23, "IdleEnergyCoef": 0.06, "ExeEnergyCoef": 0.59},
        {"NodeType": "TrustNode", "NodeName": "n8", "NodeId": 8, "MaxCpuFreq": 5, "MaxBufferSize": 322, "LocX": 23.27, "LocY": 4.23, "IdleEnergyCoef": 0.02, "ExeEnergyCoef": 0.24},
        {"NodeType": "TrustNode", "NodeName": "n9", "NodeId": 9, "MaxCpuFreq": 12, "MaxBufferSize": 196, "LocX": 99.76, "LocY": 73.87, "IdleEnergyCoef": 0.09, "ExeEnergyCoef": 0.93},
        {"NodeType": "TrustNode", "NodeName": "n10", "NodeId": 10, "MaxCpuFreq": 23, "MaxBufferSize": 326, "LocX": 62.11, "LocY": 56.11, "IdleEnergyCoef": 0.06, "ExeEnergyCoef": 0.46},
        {"NodeType": "TrustNode", "NodeName": "n11", "NodeId": 11, "MaxCpuFreq": 13, "MaxBufferSize": 181, "LocX": 53.81, "LocY": 18.49, "IdleEnergyCoef": 0.03, "ExeEnergyCoef": 0.6},
        {"NodeType": "MaliciousNode", "NodeName": "n12", "NodeId": 12, "MaxCpuFreq": 20, "MaxBufferSize": 373, "LocX": 29.4, "LocY": 46.87, "IdleEnergyCoef": 0.05, "ExeEnergyCoef": 0.54}
    ],
    "Edges": [
        {"EdgeType": "Link", "SrcNodeID": 11, "DstNodeID": 2, "Bandwidth": 125},
        {"EdgeType": "Link", "SrcNodeID": 5, "DstNodeID": 6, "Bandwidth": 87},
        {"EdgeType": "Link", "SrcNodeID": 11, "DstNodeID": 7, "Bandwidth": 103},
        {"EdgeType": "Link", "SrcNodeID": 4, "DstNodeID": 6, "Bandwidth": 200},
        {"EdgeType": "Link", "SrcNodeID": 8, "DstNodeID": 10, "Bandwidth": 65}
    ]
}

# Randomize node properties
for node in topology["Nodes"]:
    node["LocX"] = round(random.uniform(0, 100), 2)
    node["LocY"] = round(random.uniform(0, 100), 2)
    node["MaxCpuFreq"] = random.randint(5, 30)
    node["MaxBufferSize"] = random.randint(50, 400)
    node["IdleEnergyCoef"] = round(random.uniform(0.01, 0.1), 2)
    node["ExeEnergyCoef"] = round(random.uniform(0.2, 1.0), 2)

# Randomize edges while maintaining connectivity
random.shuffle(topology["Edges"])
for edge in topology["Edges"]:
    edge["Bandwidth"] = random.randint(50, 200)

# Save the randomized topology to topo.json
with open("topo.json", "w") as file:
    json.dump(topology, file, indent=4)

print("Randomized topology saved to topo.json")
