import pandas as pd

data = pd.read_csv("examples/dataset/demo3_dataset.csv")
test_tasks = list(data.iloc[:].values)

import collections

node_dict = collections.defaultdict(list)

for _, row in data.iterrows():
    node_dict[row["SrcName"]].append(row["GenerationTime"])


for node in node_dict:
    node_dict[node].sort()

print(node_dict)