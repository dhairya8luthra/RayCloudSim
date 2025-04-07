import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('hyperparameter_tuning_results.csv')

# List of hyperparameters to compare with f1
params = ['OLD_WEIGHT', 'THRESHOLD', 'lambda_task', 'lambda_time', 'lambda_base', 'ALPHA', 'BETA']

# Create subplots (adjust rows/columns as needed)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()

# Plot each hyperparameter vs. f1
for i, param in enumerate(params):
    if param in df.columns:
        axes[i].scatter(df[param], df['f1'], alpha=0.7)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('f1')
        axes[i].set_title(f'{param} vs f1')

# Remove any unused axes
for j in range(len(params), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()