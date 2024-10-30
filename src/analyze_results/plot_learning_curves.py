import numpy as np
import os
import json
import random
import string

from ..path_manager import PathManager

import matplotlib.pyplot as plt
import seaborn as sns

PRIOR_KNOWLEDGE2NAME = dict(
    decreasing_exp = "Prior knowledge: decreasing exponential",
    uniform = "Prior knowledge: uniform",
    zero = "Prior knowledge: none"
)
METHOD2NAME = dict(
    # ours_pretrained = "Ours (expert data + RL)",
    ours_pretrained = "Ours",
    vassoyan = "Vassoyan et al.",
    bassen = "Bassen et al.",
    irt = "CMAB",
    ours_pretrained_next_feedback = "Ours (next feedback)",
    ours_pretrained_expert = "Ours (expert data)"
)
LEGEND_ORDER1 = {
    "Ours": 0,
    "Vassoyan et al.": 1,
    "CMAB": 2,
    "Bassen et al.": 3
}
LEGEND_ORDER2 = {
    "Ours (expert data + RL)": 0,
    "Ours (expert data)":1,
    "Ours (next feedback)":2,
    'Vassoyan et al.':3
}

plot = 1 # 1, 2
black_mode = True
plot_style = 'fill_between' # fill_between, errorbar

stats_filename = "09-17_03-15-51__mean__bootstrap_0.95_10000"
# stats_filename = "09-16_11-30-55__mean__bootstrap_0.95_10000"

stats_filename_ext = stats_filename + '.json'
stats_filepath = os.path.join(PathManager.STATS_RESULTS, stats_filename_ext)


with open(stats_filepath, 'r') as json_file:
    results_dict = json.load(json_file)

if plot==1:
    methods_to_compare = ['ours_pretrained', 'vassoyan', 'bassen', 'irt']
    legend_order = LEGEND_ORDER1
elif plot==2:
    methods_to_compare = ['ours_pretrained', 'ours_pretrained_expert', 'ours_pretrained_next_feedback', 'vassoyan']
    legend_order = LEGEND_ORDER2
else:
    raise NotImplementedError

sns.set_style('darkgrid')
palette = sns.color_palette()

method2color = dict(ours_pretrained='#8272B2', vassoyan='#CCB974', bassen='#64B5CD', irt='#C44F51', ours_pretrained_expert='#8C564C', ours_pretrained_next_feedback='#FF7F0F')

experiment_names = ['zero', 'decreasing_exp', 'uniform']
assert set(experiment_names)==results_dict.keys()
n_experiments = len(experiment_names)

# Set up the subplots
fig, axes = plt.subplots(1, n_experiments, figsize=(5 * n_experiments, 4))

# Ensure axes is a list even if there's only one subplot
if n_experiments == 1:
    axes = [axes]

for ax, experiment_name in zip(axes, experiment_names):
    methods = results_dict[experiment_name]
    methods = {key:methods[key] for key in methods_to_compare}
    for idx, (method_name, method_data) in enumerate(methods.items()):
        training_steps = sorted([int(step) for step in method_data.keys()])
        training_steps_str = [str(step) for step in training_steps]
        central_vals = [method_data[step]['central'] for step in training_steps_str]

        if plot_style=='errorbar':
            errors = np.array([(method_data[step]['central'] - method_data[step]['lower_bound'],
                                method_data[step]['upper_bound'] - method_data[step]['central'])
                                for step in training_steps_str]).transpose()
            ax.errorbar(training_steps, central_vals, yerr=errors, label=METHOD2NAME[method_name], color=method2color[method_name], capsize=3)
        
        elif plot_style=='fill_between':
            errors = [(method_data[step]['lower_bound'], method_data[step]['upper_bound'])
                    for step in training_steps_str]
            ax.plot(training_steps, central_vals, label=METHOD2NAME[method_name], color=method2color[method_name])
            ax.fill_between(training_steps, 
                            [error[0] for error in errors], 
                            [error[1] for error in errors], 
                            color=method2color[method_name], alpha=0.3)
        else:
            raise NotImplementedError
    
    ax.set_title(PRIOR_KNOWLEDGE2NAME[experiment_name], fontsize=14)
    ax.set_xlabel('student', fontsize=12)
    ax.set_xticks(training_steps)

    if black_mode:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')


# Set y-label on the first subplot
axes[0].set_ylabel('learning gains', fontsize=12)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend

# Get handles and labels for the legend from the first axis
handles, labels = axes[0].get_legend_handles_labels()

labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: legend_order[t[0]]))

# Place the legend below the subplots
fig.legend(handles, labels, loc='lower center', ncol=len(handles), fontsize=14)

if black_mode:
    fig.patch.set_facecolor('black')
    
    # legend = axes.legend()
    # for text in legend.get_texts():
    #     text.set_color('white')


random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
plot_filename = stats_filename + '__plot_' + random_string + ".png"
plot_filepath = os.path.join(PathManager.PLOTS_RESULTS, plot_filename)
plt.savefig(plot_filepath)

print(f'Saved figure in: {plot_filepath}')