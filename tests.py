#!/usr/bin/env python3

# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os
import shutil
import sys
sys.path.insert(0, './VAE-IW')
import torch
import gym
# from gym.envs.classic_control import rendering
from gym.wrappers import Monitor
from dataset import PixelDataSet, create_test_dataset, create_data_loaders
from train_encoder import make_model, train_epoch
from active_learning_screen import ActiveLearningScreen
from active_learning_tree import ActiveLearningTreeActor
from active_rollout_iw import ActiveRolloutIW
from atari_wrapper import wrap_atari_env
from utils import env_has_wrapper, remove_env_wrapper
from rolloutIW import RolloutIW, NoveltyTable
from screen import Screen
from tree import TreeActor
from sample import softmax_Q, sample_cdf
from PIL import Image
import numpy as np
import time
import argparse


def setup_environment():
    env = gym.make('MsPacman-v0')
    env.reset()

    return env, env.action_space


def test_dataset():
    frame_skip = 15
    env = gym.make('MsPacman-v0')
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, frame_skip)

    return create_test_dataset(env, './MsPacman/', 100, 10000)


def test_alien_dataset():
    frame_skip = 15
    env = gym.make('Alien-v4')
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, frame_skip)

    return create_test_dataset(env, './Alien/', 100, 10000)


def test_train():
    dataset = PixelDataSet(dir='./MsPacman/')
    train_loader, test_loader, train_set, test_set = create_data_loaders(dataset, 64)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(1,11):
        train_epoch(model, optimizer, train_loader, i)

    return model


def test_train_alien():
    dataset = PixelDataSet(dir='./Alien/')
    train_loader, test_loader, train_set, test_set = create_data_loaders(dataset, 64)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(1,11):
        train_epoch(model, optimizer, train_loader, i)

    return model

# Save and load the model as
# torch.save(model.state_dict(), PATH)
# model = make_model()
# model.load_state_dict(torch.load(PATH))


def test_reconstruction():
    mod = make_model()
    mod.load_state_dict(torch.load('PacmanModel.pt'))
    orig = torch.load('./MsPacman/0.pt')
    orig = orig[None, :, :, :]
    enc = mod.encode(orig)
    dec = mod.decode(enc)

    orig_im = (orig * 255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    out_im = (dec * 255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    out = Image.fromarray(np.hstack((orig_im, out_im)))
    out.show()
    return out


def test_alien_reconstruction(idx):
    mod = make_model()
    mod.load_state_dict(torch.load('AlienModel.pt'))
    orig = torch.load('./Alien/'+str(idx)+'.pt')
    orig = orig[None, :, :, :]
    enc = mod.encode(orig)
    dec = mod.decode(enc)

    orig_im = (orig*255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    out_im = (dec*255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    out = Image.fromarray(np.hstack((orig_im, out_im)))

    out.show()
    return out


def rollout_and_retrain(screen, tree_actor, plan_index, budget=0.5, n_rollouts=200, n_epochs=30, start_new=False,
                        pruning='prob-IW', features='online-VAE', method='active', episode=0):
    start_time = time.time()
    # Target at least 1 rollout per action
    time_per_rollout = budget / tree_actor.env.action_space.n

    if features == 'B-PROST':
        feature_size = 20598848
    else:
        feature_size = 20 * 15 * 15

    if method == 'active':
        active_rollout_iw = ActiveRolloutIW(tree_actor.env.action_space.n, feature_size=feature_size, width=1,
                                            pruning=pruning, features=features, max_depth=100)
        rollout_start = time.time()
        active_rollout_iw.rollout(tree_actor.tree.root, start_new, plan_index,
                                  action_f=tree_actor.sample_action,
                                  successor_f=tree_actor.getSuccessor,
                                  update_index_f=tree_actor.reset_for_plan_index,
                                  update_dataset_f=tree_actor.update_dataset,
                                  stop_time_budget=lambda current_time: current_time - rollout_start > time_per_rollout or
                                                                        current_time - start_time > budget)
        i = 0
        while True:
            if time.time() - start_time > budget:
                break
            rollout_start = time.time()
            active_rollout_iw.rollout(tree_actor.tree.root, False, plan_index,
                                      action_f=tree_actor.sample_action,
                                      successor_f=tree_actor.getSuccessor,
                                      update_index_f=tree_actor.reset_for_plan_index,
                                      update_dataset_f=tree_actor.update_dataset,
                                      stop_time_budget=lambda current_time: current_time - rollout_start > time_per_rollout or
                                                                            current_time - start_time > budget)
            i += 1

        # print("Completed {} rollouts".format(i))
        # print("Max rollout depth = {}".format(tree_actor.tree.max_depth))
        # screen.retrain_model(n_epochs)

    elif method == 'rollout_iw':
        rollout_iw = RolloutIW(tree_actor.env.action_space.n, feature_size=feature_size, width=1)
        rollout_iw.novelty_table = NoveltyTable(rollout_iw.ignore_tree_caching, feature_size=rollout_iw.feature_size,
                                                width=rollout_iw.width)
        policy = lambda n, bf: np.full(bf, 1 / bf)
        rollout_iw.initialize(tree_actor.tree)

        i = 0
        while True:
            if time.time() - start_time > budget:
                break
            rollout_iw.rollout_length = 0
            a, node = rollout_iw.select(tree_actor.tree.root, policy(tree_actor.tree.root, rollout_iw.branching_factor))
            rollout_iw.rollout_length = rollout_iw.rollout_length + 1
            if a is not None:
                rollout_iw.rollout(node, a, successor_f=tree_actor.getSuccessor, stop_simulator=lambda: False,
                                   stop_time_budget=lambda current_time: current_time - start_time > budget,
                                   policy=policy)
            rollout_iw.rollouts = []
            i += 1

        # print("Completed {} rollouts".format(i))
        # print("Max rollout depth = {}".format(tree_actor.tree.max_depth))

    screen.writer.add_scalar('Gameplay/' + str(episode) + '/Rollouts', i, plan_index)
    screen.writer.add_scalar('Gameplay/' + str(episode) + '/MaxDepth', tree_actor.tree.max_depth, plan_index)


def test_alien_planning():
    method = 'active'

    if method == 'active':
        # screen = ActiveLearningScreen(zdim=20, xydim=15, datasetsize=128, data_dir='./Alien-2/',
        #                              model_name='AlienModel.pt')
        screen = ActiveLearningScreen(zdim=20, xydim=15, datasetsize=128, data_dir='./Alien-2/')
        # screen = ActiveLearningScreen(zdim=20, xydim=15, datasetsize=128, data_dir='./Alien-2/',
        #                               model_name='../VAE-IW/data/Alien/model/model.pt')
        screens_collected = len(screen.dataset)

    elif method == 'rollout_iw':
        screen = Screen(features='model', model_name='../VAE-IW/data/Alien/model/model.pt', zdim=20, xydim=15,
                        datasetsize=128)
        screens_collected = 0

    viewer = rendering.SimpleImageViewer()

    frame_skip = 15
    env = gym.make('Alien-v4')
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, frame_skip)

    if method == 'active':
        tree_actor = ActiveLearningTreeActor(env, screen.GetFeatures, screen.update_dataset,
                                             n_actions=env.action_space.n)
    elif method == 'rollout_iw':
        tree_actor = TreeActor(env, screen.GetFeatures)

    tree_actor.reset()
    # rgb = env.render('rgb_array')
    # upscaled = repeat_upsample(rgb, 4, 4)
    # viewer.imshow(upscaled)

    plan_index = 0
    retrain_per_step = 1

    screens_collected = len(screen.dataset)
    total_reward = 0
    round_reward = 0
    round_rewards = []
    episode_done = False

    while not (episode_done or env.unwrapped.ale.game_over()):
        for i in range(retrain_per_step):
            rollout_and_retrain(screen, tree_actor, plan_index, start_new=(plan_index == 0), method=method)
            plan_index += 1

        if method == 'active':
            #for a, reward in sorted(tree_actor.tree.root.Q.items(), key=lambda x: x[0]):
            #    print("a = {}, mean = {}".format(a, reward._mu))
            prev, curr = tree_actor.step(tree_actor.best_action(), render=True)

        elif method == 'rollout_iw':
            p = softmax_Q(tree_actor.tree, env.action_space.n, 0.99, tree_actor.tree.root.data["ale.lives"],
                          risk_averse=False)
            a = sample_cdf(p.cumsum())
            prev, curr = tree_actor.step(a, render=True)

        # env.unwrapped.restore_state(curr["s"])
        # # Rendering is bugged after restoring state until another step is taken, so do nothing
        # env.step(0)
        # rgb = env.render('rgb_array')
        # upscaled = repeat_upsample(rgb, 4, 4)
        # viewer.imshow(upscaled)
        # env.unwrapped.restore_state(curr["s"])
        # tree_actor.last_node = tree_actor.tree.root

        episode_done = curr["done"]
        total_reward += curr["r"]
        round_reward += curr["r"]
        print("Round reward = {}".format(round_reward))

    round_rewards += [(screens_collected, round_reward)]
    print("Round rewards = {}".format(round_rewards))
    screen.dataset.save()
    torch.cuda.empty_cache()

    if method != 'active':
        return tree_actor, screen

    for episode in range(8):
        screen.retrain_model(50)
        tree_actor.reset()
        # rgb = env.render('rgb_array')
        # upscaled = repeat_upsample(rgb, 4, 4)
        # viewer.imshow(upscaled)

        screens_collected = len(screen.dataset)
        plan_index = 0
        total_reward = 0
        round_reward = 0
        episode_done = False

        while not (episode_done or env.unwrapped.ale.game_over()):
            for i in range(retrain_per_step):
                rollout_and_retrain(screen, tree_actor, plan_index, start_new=(plan_index == 0), method=method)
                plan_index += 1

            if method == 'active':
                #for a, reward in sorted(tree_actor.tree.root.Q.items(), key=lambda x: x[0]):
                #    print("a = {}, mean = {}".format(a, reward._mu))
                prev, curr = tree_actor.step(tree_actor.best_action(), render=True)

            elif method == 'rollout_iw':
                p = softmax_Q(tree_actor.tree, env.action_space.n, 0.99, tree_actor.tree.root.data["ale.lives"],
                              risk_averse=False)
                a = sample_cdf(p.cumsum())
                prev, curr = tree_actor.step(a, render=True)

            # env.unwrapped.restore_state(curr["s"])
            # # Rendering is bugged after restoring state until another step is taken, so do nothing
            # env.step(0)
            # rgb = env.render('rgb_array')
            # upscaled = repeat_upsample(rgb, 4, 4)
            # viewer.imshow(upscaled)
            # env.unwrapped.restore_state(curr["s"])
            # tree_actor.last_node = tree_actor.tree.root

            episode_done = curr["done"]
            total_reward += curr["r"]
            round_reward += curr["r"]
            print("Round reward = {}".format(round_reward))

        round_rewards += [(screens_collected, round_reward)]
        print("Round rewards = {}".format(round_rewards))
        screen.dataset.save()

    # Trained generated 57,283 unique screens and scored 4471
    # Untrained generated 6,574 unique screens and scored 1930
    # After updating dataset generation, untrained generated 3,605 unique screens and scored 2980
    # After updating dataset generation, 2nd play generated 42,459 unique screens and scored 4060
    # Baseline got score of 6560
    # Baseline + Thompson sampling scored 9161

    # Now we have
    # Round rewards = [860.0, 2560.0, 1060.0, 2540.0, 11120.0, 4080.0, 3970.0, 8720.0]
    # Round rewards = [(0, 2460.0), (18, 3050.0), (64, 3560.0), (187, 6070.0), (319, 6070.0), (774, 9160.0),
    #                  (1529, 2910.0), (1600, 6080.0), (1764, 8080.0)]
    # Round rewards = [(0, 4010.0), (17, 2070.0), (25, 2520.0), (49, 6660.0), (76, 3560.0), (111, 4550.0),
    #                  (371, 6340.0), (726, 6010.0), (1125, 10010.0)]

    # ten = torch.load('./Alien-2/0.pt')
    # ten = ten[None, :, :, :]
    # enc = screen.model.encode(ten)
    # dec = screen.model.decode(enc)
    # ten_im = (ten*255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    # dec_im = (dec*255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    # out_im = np.hstack((ten_im, dec_im))
    #
    #
    #     enc = screen.model.encode(ten)
    #     dec = screen.model.decode(enc)
    #     dec_im = (dec * 255).squeeze(0).squeeze(0).detach().numpy().astype(np.uint8)
    #     out_im = np.hstack((out_im, dec_im))

    # out = Image.fromarray(out_im)
    # out.show()


    # enc = torch.sigmoid(enc.squeeze(0).detach())
    # print()

    return tree_actor, screen


def make_test_env():
    frame_skip = 15
    env = gym.make('Alien-v4')
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
    env = wrap_atari_env(env, frame_skip)

    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active-IW entry point',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('env', default="Alien", help='Environment name.')
    parser.add_argument('dir', default="./screens/", help='Where to put screen data.')
    parser.add_argument('out', default="./results.csv", help='File where to put results.')
    parser.add_argument('--model',
                        default=None,
                        help='If specified, load this model (network weights for VAEs produced by torch.save())'+
                        ' at the start of planning.')
    parser.add_argument('--method',
                        choices=['rollout_iw','active'],
                        default="active",
                        help='Method for planning.')
    parser.add_argument('--episodes', default=12, type=int, help='Number of episodes of planning to perform.')
    parser.add_argument('--render', action='store_true', help='Render screens during gameplay.')
    parser.add_argument('--budget', default=0.5, type=float, help='Planning time per action.')
    parser.add_argument('--loss',
                        choices=['BCE','MSE', 'BCE_low_prior'],
                        default="BCE",
                        help='Loss function to use for VAE training.')
    parser.add_argument('--sigma', default=0.1, type=float, help='Sigma to use for MSE loss function.')
    parser.add_argument('--beta', default=1.0, type=float, help='Beta to use in Beta-VAE.')
    parser.add_argument('--bandit',
                        choices=['uniform','TS','TTTS','SH','UCT'],
                        default="TTTS",
                        help='Strategy to use to select samples. '+
                        "'Choices of strategy are "+
                        # todo: add citations and references
                        "'uniform', 'TS' (Thompson Sampling), 'TTTS' (Top Two Thompson Sampling), 'SH' (successive halving), and 'UCT'.")
    parser.add_argument('--pruning',
                        choices=['none', 'rollout-IW', 'prob-IW'],
                        default="prob-IW",
                        help="Algorithm used to prune the search space.")
    parser.add_argument('--features',
                        choices=['B-PROST','VAE','online-VAE'],
                        default="online-VAE",
                        help="Method of computing features from screens.")

    args = parser.parse_args()

    round_rewards, tree_actor, screen = test_planning(args.env,
                                                      args.dir,
                                                      model_name=args.model,
                                                      method=args.method,
                                                      episodes=args.episodes,
                                                      render=args.render,
                                                      budget=args.budget,
                                                      loss=args.loss,
                                                      sigma=args.sigma,
                                                      beta=args.beta,
                                                      bandit=args.bandit,
                                                      pruning=args.pruning,
                                                      features=args.features)

    with open(args.out, 'w') as f:
        f.write(
            "screens,reward,train_loss,train_kld_loss,train_recon_loss,test_loss,test_kld_loss,test_recon_loss\n")
        for rr in round_rewards:
            f.write("{},{},{},{},{},{},{},{}\n".format(rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], rr[6], rr[7]))

