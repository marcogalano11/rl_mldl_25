import glob
import numpy as np
import matplotlib.pyplot as plt

def compute_from_folders(folder):
    txt_files = glob.glob(f"{folder}/*.txt")

    mat = []
    for file in txt_files:
        arr = np.loadtxt(file)
        mat.append(arr)
    
    values = np.mean(np.array(mat), axis=0)
    stds = np.std(np.array(mat), axis=0)

    mvg_avg = np.convolve(values, np.ones(1000) / 1000, mode='valid')
    mvg_avg_stds = np.convolve(stds, np.ones(1000) / 1000, mode='valid')

    upper = mvg_avg + mvg_avg_stds
    lower = mvg_avg - mvg_avg_stds

    return mvg_avg, upper, lower

# ACTOR CRITIC

mvg_avg, upper, lower = compute_from_folders("outputs_actor_critic")
episodes = range(1, len(mvg_avg)+1)
plt.plot(episodes, mvg_avg, color='blue', linewidth=2, label="ACTOR CRITIC")
plt.fill_between(episodes, lower, upper, color='blue', alpha=0.1)

#REINFORCE
mvg_avg, upper, lower = compute_from_folders("outputs_reinforce")
episodes = range(1, len(mvg_avg)+1)
plt.plot(episodes, mvg_avg, color='red', linewidth=2, label="REINFORCE")
plt.fill_between(episodes, lower, upper, color='red', alpha=0.1)

#REINFORCE with baseline
mvg_avg, upper, lower = compute_from_folders("outputs_reinforce_baseline")
episodes = range(1, len(mvg_avg)+1)
plt.plot(episodes, mvg_avg, color='green', linewidth=2, label="REINFORCE with baseline")
plt.fill_between(episodes, lower, upper, color='green', alpha=0.1)


plt.legend(loc='upper left', fontsize=15)
plt.ylabel("Rewards")
plt.xlabel("Episodes")
plt.show()
