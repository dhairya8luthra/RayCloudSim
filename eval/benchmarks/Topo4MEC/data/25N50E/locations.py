import json
import random

# Define latitude and longitude ranges (adjust as needed)
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

# Load the JSON file
with open("config.json", "r") as file:
    data = json.load(file)

# Assign random locations to each node
for node in data["Nodes"]:
    node["LocX"] = round(random.uniform(LON_MIN, LON_MAX), 6)  # Longitude
    node["LocY"] = round(random.uniform(LAT_MIN, LAT_MAX), 6)  # Latitude

# Save the updated JSON
with open("config.json", "w") as file:
    json.dump(data, file, indent=4)

print("Updated config.json with random locations.")
