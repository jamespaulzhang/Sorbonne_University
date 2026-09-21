import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_all_trials(prefix, num_trials):
    all_data = []
    for i in range(1, num_trials + 1):
        file_name = f"{prefix}_{i}.csv"
        df = pd.read_csv(file_name, usecols=[0, 1, 2], names=["trial", "total_score", "best_score"], header=0)
        df['run'] = i
        df['best_so_far'] = df['best_score'].cummax()
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

random_data = read_all_trials("results/random_search_results", 10)
genetic_data = read_all_trials("results/genetic_algorithm_results", 10)

for df in [random_data, genetic_data]:
    df['trial'] = df['trial'].astype(int)
    df['total_score'] = df['total_score'].astype(float)
    df['best_score'] = df['best_score'].astype(float)

def plot_combined(df, label, color):
    trials = sorted(df['trial'].unique())

    grouped = df.groupby('trial')
    data_per_trial = [grouped.get_group(t)['best_so_far'].values for t in trials]

    plt.boxplot(data_per_trial, positions=trials, widths=0.6, patch_artist=True,
                boxprops=dict(facecolor=color, color=color, alpha=0.3),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(markerfacecolor=color, marker='o', alpha=0.2, markersize=3))

    avg_per_trial = df.groupby('trial')['best_so_far'].mean()
    plt.plot(trials, avg_per_trial, label=label + ' (moyenne)', color=color, linewidth=2)

plt.figure(figsize=(14, 6))
plot_combined(random_data, 'Recherche aléatoire', color='blue')
plot_combined(genetic_data, 'Algorithme génétique', color='green')

plt.xlabel('Évaluations (échelle fois 50)')
plt.ylabel('Score du meilleur individu')
plt.title('Comparaison des performances (10 essais)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Set x-axis ticks at specific positions
plt.xticks(np.arange(50, 550, 50))

plt.tight_layout()
plt.show()