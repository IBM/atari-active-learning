#!/usr/bin/env python3

# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import sys
import os
import os.path
import pandas as pd
import itertools
import stacktrace
import glob
import json
import numpy as np
import multiprocessing as mp
from scipy.stats import mannwhitneyu
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_info_columns', 500)

def ensure_list(thing):
    if isinstance(thing, list):
        return thing
    elif isinstance(thing, tuple):
        return list(thing)
    else:
        return [thing]

def read_json(path):
    with open(path) as f:
        obj = json.load(f)
    directory, name = os.path.split(path)
    obj["dir"] = directory
    if "git" in obj:
        del obj["git"]
    if "seed" in obj:
        del obj["seed"]
    if "episode" in obj:
        del obj["episode"]
    return obj

p = mp.Pool(8)
thresholds = [0.05]
# thresholds = [0.01,0.05,0.1]

def columnize(name):
    table = os.path.splitext(name)[0]+".table"
    os.system(f"column -t -s, {name} > {table}")


def load_results(name,dirs):
    print(name)
    dirs = ensure_list(dirs)
    eval_jsons = []
    param_jsons = []
    for d in dirs:
        eval_jsons.extend(glob.glob(f"{d}/eval-*.json"))
        param_jsons.extend(glob.glob(f"{d}/parameters.json"))
    evals  = pd.DataFrame(p.map(read_json, eval_jsons))
    params = pd.DataFrame(p.map(read_json, param_jsons))
    results = pd.merge(evals,params,on="dir")
    results = results.drop(["out_dir","weights","dir"],axis="columns",errors="ignore")
    return results, params


def cols_to_drop(params, ignore):
    # extract configurations that do not change
    # https://stackoverflow.com/questions/39658574/how-to-drop-columns-which-have-same-values-in-all-rows-via-pandas-or-spark-dataf
    nunique = params.nunique()
    cols = nunique[nunique == 1].index
    # also remove unnecessary axes
    cols = cols.union(ignore)

    # for a pivot table, also ignore these additioal axes
    non_reward_columns = ["actions",
                          "elap",
                          "evaluation_simulator_calls",
                          "node_per_sec",
                          "realtime_speedup",
                          "training_simulator_calls",]
    more_cols = cols.union(non_reward_columns)
    return cols, more_cols


def dump_sec_action(name,results):
    # dump sec/action
    node_per_action = results[["budget","env_name"]].groupby("env_name").agg(["mean"])["budget"]
    node_per_sec = results[["node_per_sec","env_name"]].groupby("env_name").agg(["mean"])["node_per_sec"]
    sec_per_action = node_per_action / node_per_sec
    sec_per_action.to_csv(f"{name}-sec.csv")
    columnize(f"{name}-sec.csv")



visually_changing_envs  = [
    "BankHeist",
    "JamesBond",
    "Pitfall",
    "Venture",
    "WizardOfWor",
    "Amidar",
    "TimePilot",
    "Montezuma"
]



# note: sort in order of importance.
envs = ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
        'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout',
        'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
        'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
        'IceHockey', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster',
        'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong',
        'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
        'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
        'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']
losses = ['BCE_zerosup', 'BCE_low_prior', 'BCE_std_prior',]
prunings = ['none', 'rollout-IW-threshold', 'rollout-IW', 'prob-IW', ]
bandits = ['uniform', 'max', 'TS', 'SH', 'UCT', 'TTTS', ]
features = ['B-PROST','VAE','online-VAE']
screen_selections = ['random', 'entropy', 'rec', 'loss', 'elbo', 'portfolio']
episode_limit = [18000,1800,400,200,100]
reuse = [False,True]
agg = ["count","max","mean","std"]

strings = envs + losses + prunings + bandits + features + screen_selections + episode_limit + agg
string_map = { s : i for i,s in enumerate(strings) }

def sorter(series):
    def safe_find(x):
        try:
            return strings.index(x)
        except:
            return None
    # print(series)
    # print(series.map(lambda x: safe_find(x) or x))
    return series.map(lambda x: safe_find(x) or x)


def dump_results(name,dirs,ignore=[],envs=None,portfolio=None):
    try:
        _dump_results(name,dirs,ignore,envs,portfolio)
    except KeyboardInterrupt as e:
        raise e
    except:
        stacktrace.format(exit=False)


def _dump_results(name,dirs,ignore=[],envs=None,portfolio=None):
    results, params = load_results(name,dirs)
    cols, more_cols = cols_to_drop(params, ignore)
    if envs is not None:
        results = results[results["env_name"].isin(envs)]
        params = params[params["env_name"].isin(envs)]

    # dump all interesting metrics into a csv
    results.drop(cols,axis="columns",errors="ignore").to_csv(f"{name}-results.csv")

    # dump just rewards as a pivot table
    pd.pivot_table(results.drop(more_cols,axis="columns",errors="ignore"),
                   values="reward",
                   index=["env_name"],
                   columns=list(results.columns.difference(["reward","env_name",*list(more_cols)])),
                   aggfunc=["count","max","mean","std",])\
      .stack(0) \
      .sort_index(key=sorter,axis="columns") \
      .unstack() \
      .round(1) \
      .dropna() \
      .astype(int) \
      .to_csv(f"{name}-pivot.csv")
    columnize(f"{name}-pivot.csv")

    # dump sec/action
    dump_sec_action(name,results)

    # obtain a list of distinct configurations, ignoreing dir and env_name
    params2 = results[list(results.columns.difference(["reward","env_name",*list(more_cols)]))].drop_duplicates()
    results2 = results.drop(more_cols,axis="columns",errors="ignore")
    cols1 = [ var+"_1" for var in params2.columns]
    cols2 = [ var+"_2" for var in params2.columns]
    if params2.empty:
        return
    for pval_threshold in thresholds:
        print("pval:",pval_threshold)
        wins = []
        for i,(_,c1) in enumerate(params2.iterrows()):
            for j,(_,c2) in enumerate(params2.iterrows()):
                if i != j:
                    # print(list(c1),list(c2))
                    row = {}
                    for var,val in zip(cols1,c1):
                        row[var] = val
                    for var,val in zip(cols2,c2):
                        row[var] = val
                    row["wins"], _ = \
                        pairwise_significant_wins(
                            results2[(results2[list(params2)] == c1).all(axis="columns")],
                            results2[(results2[list(params2)] == c2).all(axis="columns")],
                            pval_threshold)
                    wins.append(row)

        if portfolio is not None:
            c1 = params2.iloc[-1]
            c2 = params2.iloc[-2]
            portfolio = c1.copy()
            portfolio["screen_selection"] = "portfolio"
            total = len(params2)
            for i,(_,c) in enumerate(params2.iterrows()):
                if i not in [total-2,total-1]:
                    row = {}
                    for var,val in zip(cols1,c):
                        row[var] = val
                    for var,val in zip(cols2,portfolio):
                        row[var] = val
                    _, win_envs1 = \
                        pairwise_significant_wins(
                            results2[(results2[list(params2)] == c).all(axis="columns")],
                            results2[(results2[list(params2)] == c1).all(axis="columns")],
                            pval_threshold)
                    _, win_envs2 = \
                        pairwise_significant_wins(
                            results2[(results2[list(params2)] == c).all(axis="columns")],
                            results2[(results2[list(params2)] == c2).all(axis="columns")],
                            pval_threshold)
                    win_envs = win_envs1 & win_envs2
                    row["wins"] = len(win_envs)
                    wins.append(row)

                    row = {}
                    for var,val in zip(cols2,c):
                        row[var] = val
                    for var,val in zip(cols1,portfolio):
                        row[var] = val
                    _, win_envs1 = \
                        pairwise_significant_wins(
                            results2[(results2[list(params2)] == c1).all(axis="columns")],
                            results2[(results2[list(params2)] == c).all(axis="columns")],
                            pval_threshold)
                    _, win_envs2 = \
                        pairwise_significant_wins(
                            results2[(results2[list(params2)] == c2).all(axis="columns")],
                            results2[(results2[list(params2)] == c).all(axis="columns")],
                            pval_threshold)
                    win_envs = win_envs1 | win_envs2
                    row["wins"] = len(win_envs)
                    wins.append(row)


        wins = pd.DataFrame(wins)
        wins = pd.pivot(wins,
                        values="wins",
                        index=cols1,
                        columns=cols2)
        wins = wins.sort_index(key=sorter,axis=0)
        wins = wins.sort_index(key=sorter,axis=1)

        wins.to_csv(f"{name}-wins-{pval_threshold}.csv")
        columnize(f"{name}-wins-{pval_threshold}.csv")

        outperform = wins - wins.T
        # outperform = wins > wins.T
        # outperform[wins.isna()] = np.nan
        # outperform[wins == wins.T] = np.nan
        outperform.to_csv(f"{name}-outperform-{pval_threshold}.csv")
        columnize(f"{name}-outperform-{pval_threshold}.csv")


def pairwise_significant_wins(results1,results2,pval_threshold):
    env_names = results1["env_name"].drop_duplicates().tolist()
    wins = 0
    win_envs = set()
    for env_name in env_names:
        sub1 = results1[results1["env_name"] == env_name]
        sub2 = results2[results2["env_name"] == env_name]
        mean1 = sub1.mean()
        mean2 = sub2.mean()
        try:
            U, pval = mannwhitneyu(sub1["reward"], sub2["reward"])
        except ValueError as e:
            print(e,env_name)
            pval = float("inf")

        if (mean1["reward"] > mean2["reward"]) and (pval < pval_threshold):
            wins += 1
            win_envs.add(env_name)

    return wins, win_envs






if __name__ == '__main__':

    dump_results(
        "tmp-bandit",
        "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_*_VAE_elbo_5.0_100_False_100_18000/*")

    dump_results(
        "tmp-offline",
        ["results/*_BCE_low_prior_0.0001_rollout-IW_uniform_B-PROST_elbo_5.0_100_False_100_18000/*",
         "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_0.5_100_False_100_18000/*",
         "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_5.0_100_False_100_18000/*",
         "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_VAE_elbo_5.0_100_False_100_18000/*",
        ],
        ["pruning","loss"])

    for i in [200]:
        dump_results(
            f"tmp-online{i}",
            ["results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_VAE_elbo_5.0_100_False_100_18000/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_random_5.0_100_False_100_18000/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_18000/*",
             f"results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_random_5.0_100_False_100_{i}/*",
             f"results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_{i}/*",
            ],
            ["pruning","loss"])

        dump_results(
            f"tmp-online{i}-full",
            [
                "results/*_BCE_low_prior_0.0001_rollout-IW_uniform_B-PROST_elbo_5.0_100_False_100_18000/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_0.5_100_False_100_18000/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_5.0_100_False_100_18000/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_VAE_elbo_5.0_100_False_100_18000/*",
                # "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_random_5.0_100_False_100_18000/*",
                # "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_18000/*",
                f"results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_random_5.0_100_False_100_{i}/*",
                f"results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_{i}/*",
            ],
            ["pruning","loss"],portfolio=True)

    dump_results(
        "tmp-budget100",
        [
             f"results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_200/*",
        ],
        ["pruning","loss"])

    dump_results(
        "tmp-budget200",
        [
             f"results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_200_200/*",
        ],
        ["pruning","loss"])

