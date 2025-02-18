import simpy
import random
import csv

# Configuration
NUM_TASKS = 100  # Adjust the number of tasks as needed
NODES = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12"]

# Generate task dataset
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
