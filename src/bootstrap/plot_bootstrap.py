import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

def plot_conf_intervals(folder_name, graph_title):
	no_pretrain = pd.read_csv("./src/bootstrap/data/%s/no_pretrain.csv" % folder_name, sep = ",")
	pretrain_rl = pd.read_csv("./src/bootstrap/data/%s/pretrain_rl.csv" % folder_name, sep = ",")
	pretrain_sl = pd.read_csv("./src/bootstrap/data/%s/pretrain_sl.csv" % folder_name, sep = ",")

	df = pd.DataFrame(columns = ["Epoch", "No Pre-training", "Pre-train RL", "Pre-train SL", "No Pre-training CF low", "Pre-train RL CF low", "Pre-train SL CF low", "No Pre-training CF high", "Pre-train RL CF high", "Pre-train SL CF high"])

	for epoch in range(10):
		no_pretrain_rs = [[]]
		pretrain_rl_rs = [[]]
		pretrain_sl_rs = [[]]

		for run in range(1, 31, 3):
			no_pretrain_rs[0].append(no_pretrain.iloc[epoch, run])
			pretrain_rl_rs[0].append(pretrain_rl.iloc[epoch, run])
			pretrain_sl_rs[0].append(pretrain_sl.iloc[epoch, run])

		no_pretrain_cf = stats.bootstrap(no_pretrain_rs, np.median, confidence_level=0.95, method='percentile', random_state = 1).confidence_interval
		pretrain_rl_cf = stats.bootstrap(pretrain_rl_rs, np.median, confidence_level=0.95, method='percentile', random_state = 1).confidence_interval
		pretrain_sl_cf = stats.bootstrap(pretrain_sl_rs, np.median, confidence_level=0.95, method='percentile', random_state = 1).confidence_interval

		print(no_pretrain_cf, pretrain_rl_cf, pretrain_sl_cf)

		new_record = pd.DataFrame([{'Epoch':epoch + 1}])
		df = pd.concat([df, new_record], ignore_index=True)

		df.iloc[epoch, df.columns.get_loc("No Pre-training")] = np.mean(no_pretrain_rs)
		df.iloc[epoch, df.columns.get_loc("Pre-train RL")] = np.mean(pretrain_rl_cf)
		df.iloc[epoch, df.columns.get_loc("Pre-train SL")] = np.mean(pretrain_sl_cf)

		df.iloc[epoch, df.columns.get_loc("No Pre-training CF low")] = no_pretrain_cf.low
		df.iloc[epoch, df.columns.get_loc("Pre-train RL CF low")] = pretrain_rl_cf.low
		df.iloc[epoch, df.columns.get_loc("Pre-train SL CF low")] = pretrain_sl_cf.low

		df.iloc[epoch, df.columns.get_loc("No Pre-training CF high")] = no_pretrain_cf.high
		df.iloc[epoch, df.columns.get_loc("Pre-train RL CF high")] = pretrain_rl_cf.high
		df.iloc[epoch, df.columns.get_loc("Pre-train SL CF high")] = pretrain_sl_cf.high

	with sns.color_palette("colorblind"):
		plt.figure(figsize = (4.854, 3))

		sns.lineplot(df.loc[:, ["No Pre-training", "Pre-train RL", "Pre-train SL"]], dashes = False)
		
		plt.fill_between(list(range(10)), df["No Pre-training CF low"].to_list(), df["No Pre-training CF high"].to_list(), alpha=.3)
		plt.fill_between(list(range(10)), df["Pre-train RL CF low"].to_list(), df["Pre-train RL CF high"].to_list(), alpha=.3)
		plt.fill_between(list(range(10)), df["Pre-train SL CF low"].to_list(), df["Pre-train SL CF high"].to_list(), alpha=.3)

		plt.title(graph_title)
		plt.ylabel("Rewards (median of runs)")
		plt.xlabel("Epoch")
		plt.show()
		# plt.savefig("plots/%s.png" % folder_name, dpi = 300, bbox_inches = "tight")

plot_conf_intervals("prior_knowledge_none", "Simulated student w/o prior knowledge")
plot_conf_intervals("prior_knowledge_decreasing_exp", "Simulated student w/ exp. decreasing prior knowledge")
plot_conf_intervals("prior_knowledge_uniform", "Simulated student w/ uniform prior knowledge")