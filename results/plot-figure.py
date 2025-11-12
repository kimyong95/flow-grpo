
# plot 3x4 grid for metric
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import re
import seaborn as sns
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

colors = sns.color_palette("hls", 8)

color_map = {
    "optimize": colors[7],

    "grpo":  colors[5],
    "tree-g": colors[1],
    "d-search": colors[2],
    "dno": colors[4],
}

zorder_map = {
    "optimize": 100,

    "grpo":  1,
    "tree-g": 1,
    "d-search": 1,
    "dno": 1,
}

cache_path = f"{dir_path}/wandb_cache.pkl"
with open(cache_path, 'rb') as f:
    wandb_cache = pickle.load(f)

algos = ["optimize", "grpo", "d-search", "dno", "tree-g"]
prompt_ids = [1,2,3,4,5,6]
run_name_format = "{algo}-prompt-align-{prompt_id}"

plot_data = {}

##################### load data ######################

score_key_map = {
    "optimize": "reward_gemini",
    "grpo":     "reward_gemini",
    "d-search": "reward_gemini",
    "dno":      "reward_gemini",
    "tree-g":   "reward_gemini",
}
max_objective_evaluations = 6400
for algo in algos:
    for prompt_id in prompt_ids:
        run_name = run_name_format.format(algo=algo, prompt_id=prompt_id)
        objective_evaluations = wandb_cache[run_name]["history"]["objective_evaluations"]
        index = objective_evaluations.index[objective_evaluations.notna() & (objective_evaluations <= max_objective_evaluations)]
        objective_evaluations = objective_evaluations[index].values
        

        rewards = wandb_cache[run_name]["history"]["reward_gemini"][index].values
        assert len(objective_evaluations) == len(rewards)
        plot_data[run_name] = {
            "x": objective_evaluations,
            "y": rewards,
        }

##################### load data ######################

###################### ploting #######################
prompt_name = {
    1: "Arranged apples",
    2: "3 o'clock",
    3: "Puppy chasing butterfly",
    4: "Puppy catch apple",
    5: "Lefthand writting",
    6: "Traffic light",
}

nrows = 2
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9))
for row in range(nrows):
    for col in range(ncols):
        ax = axes[row, col]
        prompt_id = row * ncols + col + 1

        for algo_i, algo in enumerate(algos):
            run_name = run_name_format.format(algo=algo, prompt_id=prompt_id)
            label = algo if (row == 0 and col == 0) else None
            linewidth = 1.5 if algo in ["ours","ours-ddim"] else 1.5
            if algo in ["d-search", "tree-g"]:
                linestyle='dotted'
                ax.plot(plot_data[run_name]["x"][-1], plot_data[run_name]["y"][-1], 'o', color=color_map[algo])
            else:
                linestyle='solid'

            ax.plot(plot_data[run_name]["x"] , plot_data[run_name]["y"], label=label, color=color_map[algo], linewidth=linewidth, zorder=zorder_map[algo], linestyle=linestyle)
        
        ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title(f"Task-{prompt_id}: ({prompt_name[prompt_id]})")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Objective Evaluations', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Gemini Rating', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig(f"{dir_path}/figure.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y), format='jpeg', pil_kwargs={"quality":50})
###################### ploting #######################

####################### table ########################
for algo in algos:
    average_score = 0.0
    for prompt_id in prompt_ids:
        run_name = run_name_format.format(algo=algo, prompt_id=prompt_id)
        average_score += plot_data[run_name]["y"][-1] / len(prompt_ids)
    print(f"{algo}: num objective evaluations={plot_data[run_name]['x'][-1]} score={average_score:.4f}")
####################### table ########################