import json
import random

# Define latitude and longitude ranges (adjust as needed)
LAT_MIN, LAT_MAX = 0, 100.0
LON_MIN, LON_MAX = 0, 100.0
IS_IDEAL = True

# Load the JSON file
with open("config.json", "r") as file:
    data = json.load(file)

# Assign random locations to each node
for node in data["Nodes"]:
    node["LocX"] = round(random.uniform(LON_MIN, LON_MAX), 6)  # Longitude
    node["LocY"] = round(random.uniform(LAT_MIN, LAT_MAX), 6)  # Latitude
    if(IS_IDEAL):
        
        node["MaxCpuFreq"] = 10000*node["MaxCpuFreq"]
        node["MaxBufferSize"] = 1000*node["MaxBufferSize"]
for edge in data["Edges"]:
    if(IS_IDEAL):
        edge["Bandwidth"] = 1000*edge["Bandwidth"]
    

# Save the updated JSON
with open("config.json", "w") as file:
    json.dump(data, file, indent=4)

print("Updated config.json with random locations.")
