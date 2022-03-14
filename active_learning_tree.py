# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import sys
sys.path.insert(0, './VAE-IW')
from tree import Node, Tree, TreeActor
from reward_distributions import NormalGamma
import random
import torch
from sample import softmax, sample_cdf
from collections import defaultdict
import numpy as np
from math import floor, ceil, log2
import copy


class ActiveLearningNode(Node):

    def __init__(self, data, parent=None, reward_dist='NormalGamma', novelty=1.0, n_actions=18, plan_index=0):
        self.n_actions = n_actions
        self.data = data
        self.parent = parent
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.Q = {}
        self._reward_dist = reward_dist
        self.novelty = novelty
        self._unsampled_actions = [*range(n_actions)]
        self.children = {}
        self.plan_index = plan_index

        self.budget = n_actions
        self.log2n = ceil(log2(n_actions))
        self.choice_size = n_actions
        self.n_samples = 0
        self.S_set = [*range(n_actions)]
        self.S_size = n_actions
        self.round = 0
        self.round_sample_limit = floor(self.budget/(self.S_size * self.log2n))
        self.sample_counts = {}
        for i in range(self.n_actions):
            self.sample_counts[i] = 0
        self.sample_set = [*range(n_actions)]

        self.incomplete_actions = [*range(n_actions)]

    def remove_circular_refs(self):
        for _, child in self.children.items():
            child.parent = None
            child.remove_circular_refs()

    def add_data(self, data, novelty, n_actions):
        return ActiveLearningNode(data, parent=self, reward_dist=self._reward_dist, novelty=novelty,
                                  n_actions=n_actions)

    def update_q(self, action, reward):
        # If there are still unsampled actions, remove it and update Q
        if action in self._unsampled_actions:
            self._unsampled_actions.remove(action)
            if self._reward_dist == 'NormalGamma':
                self.Q[action] = NormalGamma()
            else:
                raise NotImplementedError('Unknown reward distribution specified.')

        self.Q[action].update(reward)

    def sample_action(self, bandit):
        """Select an action to sample from each node. Choices of strategy are 'uniform', 'TS' (Thompson Sampling),
         'TTTS' (Top Two Thompson Sampling), 'SH' (successive halving), 'UCT', and 'max' (max empirical Q)."""

        if bandit == 'uniform':
            return random.choice(self.incomplete_actions)

        elif bandit == 'max':

            # If there are still unsampled actions, randomly select
            if self._unsampled_actions:
                return random.choice(self._unsampled_actions)

            # Otherwise, select the maximum Q action
            # note: softmax-sample is necessary for tiebreaking
            p = softmax([ self.Q[a]._mu for a in self.incomplete_actions ], temp=0)
            index = sample_cdf(p.cumsum())
            return self.incomplete_actions[index]

        elif bandit == 'TS' or bandit == 'TTTS':

            # If there are still unsampled actions, randomly select
            if self._unsampled_actions:
                return random.choice(self._unsampled_actions)

            # Otherwise, use a Thompson sampling strategy
            else:
                if len(self.incomplete_actions) == 1:
                    return self.incomplete_actions[0]

                best_action = None
                best_sample = None
                for action in self.incomplete_actions:
                    reward_dist = self.Q[action]
                    sample = reward_dist.sample_posterior()
                    if best_action is None or sample > best_sample:
                        best_action = action
                        best_sample = sample

                if bandit == 'TS' or random.random() <= 0.5:
                    return best_action

                # Only search for the second best up to a fixed number of times to prevent looping when there is a
                # clearly superior action
                for attempt in range(4):
                    best_action_2 = None
                    best_sample_2 = None
                    for action in self.incomplete_actions:
                        reward_dist = self.Q[action]
                        sample = reward_dist.sample_posterior()
                        if best_action_2 is None or sample > best_sample_2:
                            best_action_2 = action
                            best_sample_2 = sample

                    if best_action_2 != best_action:
                        return best_action_2

                # If a second best action couldn't be found, return the first best
                return best_action

        # TODO: It's not immediately clear how SH works with complete actions
        elif bandit == 'SH':
            # Use a successive halving strategy
            # If the assumed budget has been used, double the budget according to the 'doubling trick'
            if self.n_samples == self.budget:
                self.budget *= 2
                self.choice_size = self.n_actions
                self.n_samples = 0
                self.S_set = [*range(self.n_actions)]
                self.S_size = self.n_actions
                self.round = 0
                self.round_sample_limit = floor(self.budget / (self.S_size * self.log2n))
                for i in range(self.n_actions):
                    self.sample_counts[i] = 0
                self.sample_set = [*range(self.n_actions)]

            if self.sample_set == []:
                self.S_size = ceil(self.S_size / 2)
                self.S_set = sorted(self.S_set, key=lambda x: self.Q[x]._mu, reverse=True)
                self.S_set = self.S_set[:self.S_size]
                self.round += 1
                self.round_sample_limit += floor(self.budget / (self.S_size * self.log2n))
                self.choice_size = self.S_size
                self.sample_set = copy.copy(self.S_set)

            action_index = random.randint(0, self.choice_size-1)
            action = self.sample_set[action_index]

            self.n_samples += 1
            self.sample_counts[action] += 1
            if self.sample_counts[action] == self.round_sample_limit:
                self.sample_set.remove(action)
                self.choice_size -= 1

            return action

        elif bandit == 'UCT':
            # If there are still unsampled actions, randomly select
            if self._unsampled_actions:
                action = random.choice(self._unsampled_actions)
                self.n_samples += 1
                self.sample_counts[action] += 1
                return action

            if len(self.incomplete_actions) == 1:
                action = self.incomplete_actions[0]
                self.n_samples += 1
                self.sample_counts[action] += 1
                return action

            # UCT is derived under an assumption that all rewards are in [0,1], so we normalize by min_Q, max_Q.
            min_Q = None
            max_Q = None
            for action in self.incomplete_actions:
                reward_dist = self.Q[action]
                if min_Q is None or reward_dist._mu < min_Q:
                    min_Q = reward_dist._mu
                if max_Q is None or reward_dist._mu > max_Q:
                    max_Q = reward_dist._mu

            Q_range = max_Q - min_Q
            # If we would divide by zero, instead make all scores 0
            if Q_range == 0:
                Q_range = 1
            best_action = None
            best_ucb = None
            for action in self.incomplete_actions:
                reward_dist = self.Q[action]
                ucb = (reward_dist._mu - min_Q)/Q_range + np.sqrt(2 * np.log(self.n_samples)/self.sample_counts[action])
                if best_action is None or ucb > best_ucb:
                    best_action = action
                    best_ucb = ucb

            return best_action

        else:
            raise ValueError('Unknown bandit algorithm', bandit)

    def reset_for_plan_index(self, plan_index, n_actions=18):
        """Resets the node when being seen for the first time in a new planning cycle."""
        # Recompute depth
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        # Reset Q functions and unsampled actions
        self.Q = {}
        self.novelty = 1.0
        self._unsampled_actions = [*range(n_actions)]
        self.plan_index = plan_index

        self.budget = n_actions
        self.choice_size = n_actions
        self.n_samples = 0
        self.S_set = [*range(n_actions)]
        self.S_size = n_actions
        self.round = 0
        self.round_sample_limit = floor(self.budget / (self.S_size * self.log2n))
        self.sample_counts = {}
        for i in range(self.n_actions):
            self.sample_counts[i] = 0
        self.sample_set = [*range(n_actions)]

        self.incomplete_actions = [*range(n_actions)]

    def create_root(self):
        if not self.check_root():
            # Remove references to the unused parts of the tree
            for action, child in self.parent.children.items():
                if child == self:
                    prev_action = action
                    break
            del self.parent.children[prev_action]
            self.parent.remove_circular_refs()
            self.parent = None

            temp_depth = self.depth
            for node in self.BFS():
                node.depth -= temp_depth

    def BFS(self):
        node_self = [self]
        while len(node_self) > 0:
            children = []
            for node in node_self:
                yield node
                children.extend([child for action, child in node.children.items()])
            node_self = children


class ActiveLearningTree(Tree):

    def __init__(self, root_data, reward_dist='NormalGamma', n_actions=18, max_depth=100):
        self.n_actions = n_actions
        self.createRoot(ActiveLearningNode(root_data, parent=None, reward_dist=reward_dist,
                                           novelty=1.0, n_actions=n_actions))

    def add_node(self, parent, data):
        child = parent.add_data(data, novelty=1.0, n_actions=self.n_actions)
        self._add_node(child)
        return child

    def createRoot(self, node):
        node.create_root()
        self.root = node

        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)
        for node in self.root.BFS():
            self._add_node(node)



# note: ActiveLearningTreeActor.step is inherited from TreeActor
# it sets self.totalSimulatorCallsThisRollout , but ActiveLearningTreeActor does not use the value
# it overrides getSuccessor

class ActiveLearningTreeActor(TreeActor):

    def __init__(self, env, getfeatures, bandit='TTTS', reward_dist='NormalGamma',
                 discount = 0.99,
                 n_actions=18, max_depth=100):
        super().__init__(env, getfeatures)
        self._bandit = bandit
        self._reward_dist = reward_dist
        self._n_actions = n_actions
        self._max_depth = max_depth
        self.discount = discount

    def reset(self):
        obs = self.env.reset()
        # Scale observations to include channels and data in range [0,1]
        self.tree = ActiveLearningTree({"obs": obs, "done": False}, reward_dist=self._reward_dist,
                                       n_actions=self._n_actions, max_depth=self._max_depth)
        self._update_state(None, self.tree.root)
        return self.tree

    def getSuccessor(self, node, action, plan_index):
        child = super().getSuccessor(node, action)
        child.plan_index = plan_index
        # Add the child to the node's children
        node.children[action] = child
        return child

    def _get_next_node(self, tree, action):
        return tree.root.children[action]

    def reset_for_plan_index(self, plan_index, node, prev_node=None):
        # Recompute the node's features
        self.getfeatures(self.env, node, prev_node)
        # Reset Q and update depth
        node.reset_for_plan_index(plan_index, self._n_actions)

    def sample_action(self, node):
        """Samples an action from node with the correct sample strategy."""
        return node.sample_action(self._bandit)

    def best_action(self, risk_averse=False):
        # note: softmax-sample is necessary for tiebreaking
        p = softmax_Q(self.tree, self._n_actions, self.discount, self.tree.root.data["ale.lives"], risk_averse=risk_averse)
        best_action = sample_cdf(p.cumsum())

        return best_action


def softmax_Q(tree, branching, discount_factor, current_lives, risk_averse):
    temp = 0
    compute_return(tree, discount_factor, current_lives, risk_averse)
    Q = np.empty(branching, dtype=np.float32)
    Q.fill(-np.inf)
    for action, child in sorted(tree.root.children.items(), key=lambda x: x[0]):
        if action in tree.root.Q:
            posterior_mean = tree.root.Q[action]._mu
        else:
            posterior_mean = None
        Q[child.data["a"]] = child.data["R"]
    return softmax(Q, temp=0)


def compute_return(tree, discount_factor, current_lives, risk_averse):
    for node in tree.iter_BFS_reverse():
        if node.is_leaf():
            if node.data["ale.lives"] < current_lives:
                R = -10 * 50000 + node.data["r"] if risk_averse else node.data["r"]
            elif node.data["r"] < 0:
                R = node.data["r"] * 50000 if node.data["r"] else node.data["r"]
            else:
                R = node.data["r"]
        else:
            if node.data["ale.lives"] < current_lives:
                cost = -10 * 50000 if risk_averse else 0
                R = cost + node.data["r"] + discount_factor * np.max([child.data["R"] for action, child in node.children.items()])
            elif node.data["r"] < 0:
                cost = node.data["r"] * 50000 if risk_averse else node.data["r"]
                R = cost + discount_factor * np.max([child.data["R"] for action, child in node.children.items()])
            else:
                R = node.data["r"] + discount_factor * np.max([child.data["R"] for action, child in node.children.items()])
        # Whether or not we commit to an action that has been searched does not depend on its novelty
        node.data["R"] = R
