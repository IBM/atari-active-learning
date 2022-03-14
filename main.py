#!/usr/bin/env python3 -u

# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os
import os.path
import shutil
import sys
sys.path.insert(0, './VAE-IW')
import torch
import gym
from active_learning_screen import ActiveLearningScreen, default_loss
from active_learning_tree import ActiveLearningTreeActor
from active_rollout_iw import ActiveRolloutIW
from atari_wrapper import wrap_atari_env
from utils import env_has_wrapper, remove_env_wrapper
from screen import Screen
from tree import TreeActor
import numpy as np
import time
import argparse
import json
import stacktrace
from printing_util import number_has_leading_zeros_p
import gc
import subprocess
from signal_util import SignalInterrupt

# definitions:
#
# * episode == A period which starts at the reset of the game, and ends with the timelimit, reaching the goal, or
#              running out of lives.
#              Between every episode the simulator is reset.
#              An episode contains multiple steps.
#
# * action  == A period between taking an action in the simulator.
#              The action is a real decision made, and there is no way back once the action is taken.
#              The search tree is cached and truncated at the root.
#              An action contains multiple rollouts.
#
# * rollout == A lookahead performed from the current state in order to collect information about the future.
#              Multiple rollouts form a search tree.
#              In this program, each rollout also has an access to the simulator.
#              A rollout contains multiple rollout steps.
#
# * rollout step == An exploratory decision made at each step in the rollout.
#                   This is governed by active learning / bandit strategy, such as UCB, TS, TTTS, HS.

# argument preprocessing

# The size of the input image after the downsampling. When the value is 128, the
# resulting image is a 128x128 monochrome image. The original size is 210 x 160 x 3.
# The value is same as the one used in VAE-IW.
datasetsize = 128
# In VAE-IW, the latent space also looks like an image. (See Res_3d_Conv_15 in VAE-IW/vae/models.py)
# xydim is the x, y dimension of the image (same value).
# zdim is the channel dimension.
# The value of xydim depends on the datasetsize (screen size) and the number of convolutional layers
# because it is decided by how many times maxpool or strided convolution is applied.
# Hence, this value is not customizable.
xydim = 15

def _feature_size(features,zdim):
    if features == 'B-PROST':
        return 20598848
    else:
        return zdim * xydim * xydim


def _start_episode(x,pattern):
    # Determine the start episode based on the files that exist
    import glob
    if x.out_dir:
        e = len(glob.glob(os.path.join(x.out_dir, pattern)))
        if e > 0:
            iprint(f"resuming from episode {e}")
        return e
    else:
        return 0


def _make_env_main(x):
    env = gym.make(x.env_name+'-v4')
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, x.frameskip)
    return env


def _setup_random_seed(x):
    import numpy.random
    numpy.random.seed(x.seed)
    import random
    random.seed(x.seed)
    import torch.random
    torch.random.manual_seed(x.seed)


# indented print
import textwrap
from colors import red, blue, yellow
meta = argparse.Namespace(counter=0,stack=[time.time()])
def iprint(msg="",increment=0,show_time=None):
    if increment < 0:
        meta.counter += increment
        last = meta.stack.pop()
        now = time.time()
        diff = now - last
        msg = red(msg)
        if show_time == None:
            show_time = True
        if show_time:
            print(blue("| "*(meta.counter-1)),msg,yellow("[{:.2f} sec]".format(diff)),flush=True)
        else:
            print(blue("| "*(meta.counter-1)),msg,flush=True)
        return diff
    elif increment==0:
        last = meta.stack[-1]
        now = time.time()
        diff = now - last
        if show_time == None:
            show_time = False
        if show_time:
            print(blue("| "*(meta.counter-1)),msg,yellow("[{:.2f} sec]".format(diff)),flush=True)
        else:
            print(blue("| "*(meta.counter-1)),msg,flush=True)
        return diff
    else:
        msg = red(msg)
        if show_time == None:
            show_time = False
        if show_time:
            print(blue("| "*(meta.counter-1)),msg,yellow("[0.0 sec]"),flush=True)
        else:
            print(blue("| "*(meta.counter-1)),msg,flush=True)
        meta.counter += increment
        now = time.time()
        meta.stack.append(now)
        return 0.0


def write_log(x : argparse.Namespace, name, **kwargs):
    # output episode-wise results
    iprint(f"writing log : {name} -- {kwargs}")
    with open(os.path.join(x.out_dir, f'{name}.json'), 'w') as f:
        json.dump(kwargs,f,indent=2)



# main procedures

def setup(x : argparse.Namespace, load_model=False, load_dataset=False):
    iprint("setup",1)

    iprint("making screen")
    screen = ActiveLearningScreen(feature_size = _feature_size(x.features,x.zdim),
                                  xydim = xydim,
                                  datasetsize  = datasetsize,
                                  **vars(x))
    if load_dataset:
        iprint(f"loading dataset")
        screen.dataset.load()
        iprint(f"loading dataset done : {len(screen.dataset)} screens")
    if load_model:
        iprint(f"loading VAE")
        screen.load_model(x.weights)
        iprint(f"loading VAE done")
    iprint(f"making screen done")

    iprint("making env")
    env = _make_env_main(x)
    iprint("making env done")

    iprint("making actor")
    tree_actor = ActiveLearningTreeActor(env, screen.GetFeatures,
                                         n_actions = env.action_space.n,
                                         discount  = x.discount,
                                         bandit    = x.bandit,)
    iprint("making actor done")

    iprint("setup done",-1)
    return screen, env, tree_actor


def main(x : argparse.Namespace):
    iprint(f"main, features = {x.features}",1)
    if x.out_dir:
        os.makedirs(x.out_dir,exist_ok=True)
    write_log(x, f'parameters', **vars(x))

    # training
    x.evaluation_phase = False
    if os.path.exists(x.weights):
        iprint(f"weight exists; skipping")
        pass

    elif x.features == "online-VAE":
        y = argparse.Namespace(**vars(x))
        # use 1/30 of training budget for initial episode (for 15000 total screens, 500 per episode)
        y.total_training_budget = x.total_training_budget // (x.screen_dataset_limit // x.screen_per_episode)
        simulator_calls = VAE_dataset_collection(y)
        screen, env, tree_actor = setup(x, load_dataset=True)

        iprint(f"episode 0 training VAE",1)
        loss = screen.retrain_model(x.epochs)
        elap = iprint(f"episode 0 training VAE done",-1)
        write_log(x, f'loss-0', episode=0, training_simulator_calls=simulator_calls, screens=len(screen.dataset), elap=elap, **loss)

        iprint(f"saving VAE",+1)
        screen.save_model(x.weights+f"_tmp0",overwrite=True)
        iprint(f"saving VAE done",-1)

        del screen, env, tree_actor
        train_online_VAE(x, simulator_calls)
        screen, env, tree_actor = setup(x, load_model=True)

        # cleanup
        if not x.keep_screen_dataset:
            iprint(f"removing dataset")
            screen.dataset.delete()

        sys.exit(3)             # training finished

    elif x.features == "VAE":
        simulator_calls = VAE_dataset_collection(argparse.Namespace(**vars(x)))
        screen, env, tree_actor = setup(x, load_dataset=True)

        iprint(f"episode 0 training VAE",1)
        loss = screen.retrain_model(x.epochs)
        elap = iprint(f"episode 0 training VAE done",-1)
        write_log(x, f'loss-0', episode=0, training_simulator_calls=simulator_calls, screens=len(screen.dataset), elap=elap, **loss)

        iprint(f"saving VAE")
        screen.save_model(x.weights)
        iprint(f"saving VAE done")
        # cleanup
        if not x.keep_screen_dataset:
            iprint(f"removing dataset")
            screen.dataset.delete()

        sys.exit(3)             # training finished

    elif x.features == "B-PROST":
        iprint(f"no training performed")
        pass
    else:
        assert False


    # evaluation
    x.evaluation_phase = True
    iprint(f"evaluation",1)
    first = True
    for i in range(_start_episode(x, f"eval-*.json"), x.evals):
        screen, env, tree_actor = setup(x, load_model=(x.features in ["VAE","online-VAE"]))
        gc.collect()
        try:
            actions, reward, evaluation_simulator_calls, elap = episode(x, 0, screen, env, tree_actor, i)
        except SignalInterrupt as e:
            # note: memory limit will not trigger this interrupt. only time limit does.
            if first == True:
                # the runtime is too short: Could not even finish a single episode.
                print("Killed before reaching the first iteration!")
                sys.exit(4)
            else:
                # pretend the standard exit status
                sys.exit(140)

        first = False
        # note : evaluation_calls is equivalent to the number of nodes visited
        write_log(x, f"eval-{i}", episode=i,
                  reward=reward, actions=actions, elap=elap,
                  evaluation_simulator_calls=evaluation_simulator_calls,
                  node_per_sec=evaluation_simulator_calls/elap,
                  realtime_speedup=(elap/actions)/(x.frameskip/60))
    iprint(f"evaluation done",-1)
    write_log(x, "done")
    iprint(f"main done",-1)
    return


def VAE_dataset_collection(x : argparse.Namespace):
    iprint(f"dataset collection start : total_training_budget = {x.total_training_budget}",1)
    target_features = x.features
    x.pruning = 'rollout-IW'
    x.features = 'B-PROST'
    if target_features == 'online-VAE':
        # collect maximum 1 epoch worth of screens
        x.screen_dataset_limit = x.screen_per_episode

    screen, env, tree_actor = setup(x)

    iprint(f"dataset collection episodes start",1)
    if target_features == 'online-VAE': # run just one episode
        _, _, simulator_calls, _ = episode(x, 0, screen, env, tree_actor, 0)
        iprint(f"adding queued nodes to the dataset",+1)
        count = screen.store_all_queued_screens()
        iprint(f"adding queued nodes to the dataset done : {count} screens added",-1)

    elif target_features == 'VAE': # run until the limit is reached
        simulator_calls = 0
        e = -1
        while simulator_calls < x.total_training_budget:
            e += 1
            _, _, simulator_calls, _ = episode(x, simulator_calls, screen, env, tree_actor, e)
            iprint(f"adding queued nodes to the dataset",+1)
            count = screen.store_all_queued_screens()
            iprint(f"adding queued nodes to the dataset done : {count} screens added",-1)

    else:
        assert False
    iprint(f"dataset collection episodes done",-1)

    iprint(f"saving dataset : {len(screen.dataset)} images",1)
    screen.dataset.save()
    iprint(f"saving dataset done",-1)


    # Delete the out_dir/runs folder which contains metrics.
    # Now contains image data only
    shutil.rmtree(os.path.join(x.out_dir,'runs'))
    iprint("runs removed")
    iprint(f"dataset collection done : simulator_calls = {simulator_calls}, total_training_budget = {x.total_training_budget}",-1)
    return simulator_calls


def train_online_VAE(x, simulator_calls):
    iprint(f"start training online VAE, simulator_calls = {simulator_calls}, total_training_budget = {x.total_training_budget}",1)
    assert not x.evaluation_phase
    e = 0
    while simulator_calls < x.total_training_budget:
        gc.collect()
        screen, env, tree_actor = setup(x, load_dataset=True)
        iprint(f"loading VAE")
        screen.load_model(x.weights+f"_tmp{e}")
        iprint(f"loading VAE done")

        e += 1
        prev_simulator_calls = simulator_calls
        actions, reward, simulator_calls, elap = episode(x, simulator_calls, screen, env, tree_actor, e)

        iprint(f"adding queued nodes to the dataset",+1)
        if simulator_calls < x.total_training_budget:
            count = screen.store_selected_queued_screens()
        else:
            count = screen.store_all_queued_screens()
        iprint(f"adding queued nodes to the dataset done : {count} screens added",-1)
        iprint(f"saving dataset : {len(screen.dataset)} images",1)
        screen.dataset.save()   # this writes to the disk
        iprint(f"saving dataset done",-1)
        torch.cuda.empty_cache()

        # tensorboard
        screen.writer.add_scalar('Gameplay/Reward',  reward,              e)
        screen.writer.add_scalar('Gameplay/Screens', len(screen.dataset), e)
        screen.writer.add_scalar('Gameplay/Actions', actions,             e)
        screen.writer.add_scalar('Gameplay/Calls',   simulator_calls,     e)

        write_log(x, f'train-{e}', episode=e, training_simulator_calls=simulator_calls, reward=reward, actions=actions,
                  elap=elap,
                  node_per_sec=(prev_simulator_calls-simulator_calls)/elap,
                  realtime_speedup=(elap/actions)/(x.frameskip/60))

        if not x.reuse_model:
            screen, env, tree_actor = setup(x, load_dataset=True)

        iprint(f"episode {e} training VAE",1)
        loss = screen.retrain_model(x.epochs)
        elap = iprint(f"episode {e} training VAE done",-1)
        write_log(x, f'loss-{e}', episode=e, training_simulator_calls=simulator_calls, screens=len(screen.dataset), elap=elap, **loss)

        iprint(f"saving VAE",+1)
        screen.save_model(x.weights+f"_tmp{e}",overwrite=True)
        iprint(f"saving VAE done",-1)

    iprint(f"training online VAE done, simulator_calls = {simulator_calls}, total_training_budget = {x.total_training_budget}",-1)
    iprint(f"saving VAE")
    screen.save_model(x.weights)
    iprint(f"saving VAE done")
    return simulator_calls


def episode(x, simulator_calls, screen, env, tree_actor, e):
    iprint(f"episode {e} planning, simulator_calls = {simulator_calls}",1)
    tree_actor.reset()
    actions = 0
    reward = 0
    episode_done = False

    if x.evaluation_phase:
        max_episode_length = x.max_episode_length
    else:
        max_episode_length = x.max_episode_length_during_training
    while not (episode_done or actions > max_episode_length):

        if x.evaluation_phase:
            simulator_calls = plan(x, screen, tree_actor, actions, start_new=(actions == 0),
                                   prev_simulator_calls = simulator_calls,
                                   simulator_limit      = simulator_calls+x.budget,
                                   episode              = e,)

        else: # training phase.
            simulator_limit = min(x.total_training_budget, simulator_calls+x.budget)
            if simulator_calls < simulator_limit:
                simulator_calls = plan(x, screen, tree_actor, actions, start_new=(actions == 0),
                                       prev_simulator_calls = simulator_calls,
                                       simulator_limit      = simulator_limit,
                                       episode              = e,)
            else:
                if number_has_leading_zeros_p(actions+1):
                    iprint(f"run out of budget")
                # the budget has run out, but we can still continue taking the best known action in the remaining tree.
                pass

            # this queues the nodes for later storage into the dataset.
            screen.queue_screens(tree_actor.tree.nodes)

        best_action = tree_actor.best_action(risk_averse=x.risk_averse)
        prev, curr = tree_actor.step(best_action, render=x.render)
        actions += 1
        reward += curr["r"]

        if x.evaluation_phase:
            episode_done = curr["done"]
        else:
            # Episode is done if a game ending state is reached or the simulator limit is reached and no children
            # remain to explore. If there are still children, we decide to explore more, but without additional planning.
            episode_done = curr["done"] or (simulator_calls >= x.total_training_budget
                                            and not tree_actor.tree.root.children)
        if number_has_leading_zeros_p(actions):
             iprint(f"actions={actions}, accumulated reward={reward}, budget used (== num.nodes, simulator calls) ={simulator_calls}", show_time=True)

    elap = iprint(f"episode {e} planning done",-1)
    return actions, reward, simulator_calls, elap


def plan(x : argparse.Namespace, screen:Screen, tree_actor:TreeActor,
         plan_index:int, prev_simulator_calls:int=0, simulator_limit=float('inf'),start_new:bool=False, episode:int=0):
    """Performs rollouts from the current root node of tree_actor until a budget is reached.

    plan_index: A counter used to reset rollout information. When this changes, variables used for UCB, TS, etc are reinitialized.
    prev_simulator_calls: The number of simulator calls already used at the start of the algorithm.
    start_new: If True, the root node is new in the tree, and its features are computed."""

    active_rollout_iw = ActiveRolloutIW(tree_actor.env.action_space.n,
                                        feature_size    = _feature_size(x.features,x.zdim),
                                        width           = 1,
                                        pruning         = x.pruning,
                                        features        = x.features,
                                        max_depth       = x.max_rollout_length,
                                        zero_novel      = x.zero_novel,
                                        discount        = x.discount,
                                        simulator_calls = prev_simulator_calls,
                                        simulator_limit = simulator_limit)

    i = 0
    while active_rollout_iw.simulator_calls < simulator_limit:
        active_rollout_iw.rollout(tree_actor.tree.root, start_new, plan_index,
                                  action_f=tree_actor.sample_action,
                                  successor_f=tree_actor.getSuccessor,
                                  update_index_f=tree_actor.reset_for_plan_index)
        if not tree_actor.tree.root.incomplete_actions:
            break
        start_new = False       # in the second or later rollouts
        i += 1

    if screen.writer:
        screen.writer.add_scalar('Gameplay/' + str(episode) + '/NodesReached',
                                 active_rollout_iw.nodes_reached,
                                 plan_index)
        screen.writer.add_scalar('Gameplay/' + str(episode) + '/AverageNovelty',
                                 active_rollout_iw.cumulative_novelty / active_rollout_iw.nodes_reached,
                                 plan_index)
        screen.writer.add_scalar('Gameplay/' + str(episode) + '/Rollouts', i, plan_index)
        screen.writer.add_scalar('Gameplay/' + str(episode) + '/MaxDepth', tree_actor.tree.max_depth, plan_index)

    # Return the final number of simulator calls
    return active_rollout_iw.simulator_calls



# command line interface

def initialize_parser():

    envs = ['adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
            'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
            'Centipede', 'ChopperCommand', 'CrazyClimber', 'defender', 'DemonAttack', 'DoubleDunk',
            'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
            'hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
            'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
            'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
            'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
            'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']

    parser = argparse.ArgumentParser(description='Active-IW entry point',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('env_name', default="Alien", help=f'Environment name. One of: {", ".join(envs)}', choices=envs, metavar='env_name')
    parser.add_argument('out_dir', default="results", help='Directory for saving the results. Created if missing.')

    parser.add_argument('--seed', default=2021, type=int, help='Random seed.')
    parser.add_argument('--evals', default=10, type=int, help='Number of evaluation runs.')

    # simulator settings
    parser.add_argument('--frameskip', default=15, type=int, help='Number of frames the agent continues pressing the same action.')
    parser.add_argument('--risk-averse', default=True, type=eval, help='Applies a risk-averse reward setting in Dittadi et al.')

    parser.add_argument('--discount', default=0.99, type=float, help='Discount factor.')

    # planning budget/resouces
    parser.add_argument('--budget', default=100, type=float,
                        help='The number of simulator calls allowed for planning between taking actions. '+
                        'The default value follows Junyent et al ICAPS2021 (pi-IW+, pi-HIW). '+
                        'The same value is used both during training (dataset collection) and evaluation. ')
    parser.add_argument('--total-training-budget', default=1e5, type=float,
                        help='The number of simulator calls allowed during the entire training (dataset collection). '+
                        'Junyent et al ICAPS2019 (pi-IW) used 40M (4e7) interactions, and '+
                        'Junyent et al ICAPS2021 (pi-IW+, pi-HIW) used 2e7 interactions (20M). '
                        'In our program, it took 9 hours to perform 2600000 = 2e6 interactions on a Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz machine. '+
                        '2e7 interactions would require 90 hours of training, which is prohibitively expensive. '+
                        'Therefore we decided to reduce it significantly to 1e5 (approx. 24 min). (we collect only 15000 = 1.5e4 images anyways.) ')
    parser.add_argument('--max-episode-length', default=18000, type=int,
                        help='Maximum number of actions that the agent can perform in each episode during evaluation. '+
                        'This essentially sets the upper bound on how long an agent plays the game, '+
                        'otherwise in some games the agent can continue playing forever. '+
                        'While this affects both the training and the testing (planning), '+
                        'note that it may take a very long time to perform 18000 steps (e.g., 12 hours). '+
                        'The default value 18000 follows Lipovetzky 2015, Sherifman 2016, Jinnai 2017, Bandles 2018, Junyent 2019;2021. '+
                        'Only VAE-IW by Dittaldi 2021 does not follow this value, using 15000.')
    parser.add_argument('--max-episode-length-during-training', default=18000, type=int,
                        help='Maximum number of actions that the agent can perform in each episode during data collection. '+
                        'See --max-episode-length.')
    parser.add_argument('--screen-dataset-limit', default=15000, type=int,
                        help='The total number of screens to be collected for VAE training. '+
                        'When features=VAE, this many screens are collected by B-PROST + Rollout IW. '+
                        'When features=Online-VAE, this is the maximum number over all episodes, '+
                        'where the maximum number of screens to be collected in each episode is specified by --screen-per-episode.')
    parser.add_argument('--max-rollout-length',
                        default=100,
                        type=int,
                        help='Maximum depth of each rollout. '+
                        'It is applied in Prob-IW, but not in Rollout-IW. '+
                        'This limit is important in Prob-IW, because it is possible that the rollout continues indefinitely. '+
                        'In contrast, in Rollout-IW, it is guaranteed to terminate without such a limit. ')

    # VAE options
    parser.add_argument('--weights',
                        default=None,
                        help='A filename of network weights for VAEs produced by torch.save(). '+
                        'If specified, and if the file exist, it skips the data collection phase and the training, '+
                        'load the model, then immediately start the evaluation. '+
                        'Meaningful only in VAE and online-VAE. ')
    parser.add_argument('--loss',
                        choices=['BCE_zerosup', 'BCE_low_prior', 'BCE_std_prior',
                                 'MSE_zerosup', 'MSE_low_prior', 'MSE_std_prior',],
                        default="BCE_std_prior",
                        help='Loss function to use for VAE training. '+
                        'BCE_zerosup follows (Asai, Kajino 2019) which minimizes the sampled latent vector z toward 0. '+
                        'It is the default of VAE-IW, but the loss is ad-hoc according to (Asai, Kajino, Fukunaga, Muise 2021). '+
                        'BCE_std_prior is a standard VAE loss for binary-concrete VAE. '+
                        'BCE_low_prior is a non-adhoc version of BCE_zerosup proposed in (Asai, Kajino, Fukunaga, Muise 2021) '+
                        'that provides a correct ELBO similar to std_prior, while still moves the sampled latent vector z toward 0. '+
                        '\n\n'+
                        'Variants starting with BCE use Binary Cross Entropy for monochrome images between [0,1], '+
                        'while those starting with MSE use square errors for datasets normalized to mean 0, variance 1. '+
                        'The log likelihood is computed with the variance specified by --sigma .')
    parser.add_argument('--epochs', default=100, type=int, help='How many epochs to train the network each time it is trained. '+
                        'Note that this is repeated between each episode in online VAE and not in offline VAE (VAE-IW), '+
                        'The default value, 100, follows VAE-IW source code README. (not mentioned in their paper.)')
    parser.add_argument('--sigma', default=0.1, type=float, help='Sigma to use for MSE loss function.')
    parser.add_argument('--beta', default=1.0, type=float, help='Beta to use in Beta-VAE.')
    parser.add_argument('--min-temperature',
                        default=0.5,
                        type=float,
                        help='Minimum temperature to use for Gumbel-Softmax. Set min/max to the same value to disable annealing.')
    parser.add_argument('--max-temperature',
                        default=5.0,
                        type=float,
                        help='Maximum temperature to use for Gumbel-Softmax. Set min/max to the same value to disable annealing.')
    parser.add_argument('--zdim', default=20, type=float, help='Latent space size. Follows the value in VAE-IW.')
    parser.add_argument('--reuse-model', action='store_true', help='Do not initialize VAE weights after each episode.')

    # rollout options
    parser.add_argument('--pruning',
                        choices=['none', 'rollout-IW', 'prob-IW', 'rollout-IW-threshold'],
                        default="prob-IW",
                        help=("Algorithm used to prune the search space. 'none' performs no pruning. "
                              "The difference between rollout-IW and rollout-IW-threshold is that "
                              "rollout-IW-threshold uses a step function on the belief state "
                              "with a threshold specified by --threshold, by default 0.9. "
                              "This is the method used in the VAE-IW paper. "
                              "I believe the threshold should 0.5, but I understand this has to do with the zerosuppress loss used in their implementation. "
                              "In contrast, rollout-IW samples a value from the belief state to compute the features. "))
    parser.add_argument('--bandit',
                        choices=['uniform', 'TS', 'TTTS', 'SH', 'UCT', 'max'],
                        default="TTTS",
                        help='Strategy to select actions during the rollout. Choices of strategy are '+
                        # todo: add citations and references
                        "'uniform', 'TS' (Thompson Sampling), 'TTTS' (Top Two Thompson Sampling), 'SH' (successive halving),'UCT', and 'max' (max Q). "+
                        "Note that this does not affect the action selection at the root node, which always selects the action with the highest mean.")
    parser.add_argument('--threshold',
                        default=0.9,
                        type=float,
                        help=("Threshold value used by --pruning rollout-IW-threshold. See documentation for --pruning."))

    # iw options
    parser.add_argument('--features',
                        choices=['B-PROST','VAE','online-VAE'],
                        default="online-VAE",
                        help="Method of computing features from screens.")
    parser.add_argument('--zero-novel',
                        default=False,
                        type=eval,
                        help='Treat a feature with value zero as novel if not seen. '+
                        'This option is hugely inefficient in BPROST+rollout-IW because the number of features is very large.')

    # active learning options
    parser.add_argument('--screen-selection',
                        choices=['entropy', 'elbo', 'loss', 'rec', 'random'],
                        default='elbo',
                        help=('Criteria for choosing the new frames to add to the dataset. '
                              'entropy: belief state entropy. '
                              'rec:     reconstruction loss. '
                              'elbo:    loss function with beta = 1. Note that this is affected by the choice of priors (see --loss). '
                              'loss:    loss function with the beta as is used in the training. '
                              'random:  New screens are selected at random. This corresponds to a passive learning scenario.'))
    parser.add_argument('--screen-per-episode',
                        default=500,
                        type=int,
                        help='The maximum number of images to be added to the dataset by active learning in each episode.')

    # misc
    parser.add_argument('--render', action='store_true', help='Opens a new GUI window and draws the screen during gameplay.')
    parser.add_argument('--keep-screen-dataset', action='store_true',
                        help='Upon completion, by default we remove the screen dataset (which could be ~1GB each) to save the disk space. '+
                        'This option disables the removal.')

    return parser


if __name__ == '__main__':

    import sys
    print(sys.argv)
    parser = initialize_parser()
    args = parser.parse_args()
    _setup_random_seed(args)
    if (args.features == "B-PROST" and args.pruning == "prob-IW"):
        print("B-PROST and prob-IW are incompatible. Falling back to rollout-IW")
        args.pruning = "rollout-IW"
    if (args.features in ['VAE','online-VAE'] and args.pruning == "none"):
        print("There is no point of learning a VAE when we don't use pruning. Falling back to B-PROST")
        args.features = "B-PROST"
    args.git = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf8").split("\n")[0]
    if args.loss == "BCE_zerosup":
        sys.exit(1)
    if args.features != "B-PROST":
        if args.pruning != 'rollout-IW-threshold':
            sys.exit(1)
    if args.features == "B-PROST":
        if args.bandit != 'uniform':
            sys.exit(1)
    print(json.dumps(vars(args),indent=2))
    try:
        main(args)
    except SignalInterrupt as e:
        sys.exit(140)
    except Exception as e:
        stacktrace.format(exit=False)
        raise e
    except SystemExit as e:
        print("exit")
        raise e




