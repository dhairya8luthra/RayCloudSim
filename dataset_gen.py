import simpy
import random
import csv

# Configuration
NUM_TASKS = 100
NODES = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12"]
ONLINE_NODES = []
ACTIVE_NODES = []

def task_generator(env, writer):
    task_id = 0
    while task_id < NUM_TASKS:
        pass
    """
    For each timestamp
    -> All the nodes inintially are inactive

    -> Gradually they become more active

    Generation algorithm:
    -> Generate the up times and the down times which describe which nodes are going go online and offline at each timestamp

    -> Nodes that are actively doing a computation should not be turned off

    -> Write the up nodes and down nodes to a json file into the format:
    Example:
    {
        42.00: {
            up_nodes: [n1, n2, n3],
            down_nodes: [n4, n5]
        },
    }

    Where 42.00 is the timestamp

    -> For each timestamp, update the ONLINE_NODES list based on the new generated up times and the down times

    -> Generate task dataset whose nodes are selected from the ONLINE_NODES list and then task is added to the dataset.
    
    """
    pass

def task_generator(env, writer):
    task_id = 0
    while task_id < NUM_TASKS:
        yield env.timeout(random.randint(1, 5))  # Random inter-arrival time
        
        task_name = f"t{task_id}"
        generation_time = env.now
        task_size = random.randint(10, 100)  # Random task size
        cycles_per_bit = random.randint(1, 10)  # Random computation requirement
        trans_bit_rate = random.randint(20, 100)  # Transmission rate
        ddl = random.randint(50, 100)  # Deadline
        src = random.choice(NODES)
        dst = random.choice([node for node in NODES if node != src])
        
        writer.writerow([task_name, generation_time, task_id, task_size, cycles_per_bit, trans_bit_rate, ddl, src, dst])
        task_id += 1

# Run simulation and write CSV
def main():
    env = simpy.Environment()
    
    with open("examples/dataset/task_dataset.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["TaskName", "GenerationTime", "TaskID", "TaskSize", "CyclesPerBit", "TransBitRate", "DDL", "SrcName", "DstName"])
        env.process(task_generator(env, writer))
        env.run()
    
    print("Dataset generated: task_dataset.csv")

if __name__ == "__main__":
    main()
