# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import sys
sys.path.insert(0, './VAE-IW')
from rolloutIW import NoveltyTable
from depth_novelty import BeliefStateDepthDataBase
import random
import time
import numpy as np


class ActiveRolloutIW:
    """This class is responsible for performing rollouts. Rollouts can be performed using a specified search algorithm,
    that may be one of 'none', 'rollout-IW', or 'prob-IW'. If 'none', it behaves like an MCTS because there is no pruning."""

    def __init__(self, branching_factor, feature_size, pruning='prob-IW', features='online-VAE',
                 width=1, max_depth=50, zero_novel=False, simulator_calls=0, simulator_limit=float('inf'), discount=0.99):
        assert width == 1  # For now, only consider width 1

        self.branching_factor = branching_factor
        self.nodes_reached = 0
        self.cumulative_novelty = 0
        self.simulator_calls = simulator_calls
        self.simulator_limit = simulator_limit

        self.features = features
        self.discount = discount
        self.max_depth = max_depth
        self.pruning = pruning
        # Use the NoveltyTable class from VAE-IW for deterministic features. It is more efficient than computing
        # probabilistic novelty.
        if pruning == 'none':
            pass
        if pruning == 'prob-IW':
            self._novelty_db = BeliefStateDepthDataBase(max_depth=max_depth, zdim=feature_size, width=width,
                                                        zero_novel=zero_novel)
        if pruning in ['rollout-IW', 'rollout-IW-threshold']:
            if zero_novel:
                self._novelty_db = NoveltyTable(True, 2*feature_size, width)
            else:
                self._novelty_db = NoveltyTable(True, feature_size, width)

    def rollout(self, node, is_new, plan_index, action_f, successor_f, update_index_f):
        """Rollout from a node. Returns a sample of Q(s,a)."""
        # If this node is from a different plan index, recompute the features, update the depth, and reset Q
        prev_index = node.plan_index
        if node.plan_index != plan_index:
            update_index_f(plan_index, node, node.parent)

        # If this node is new, or from a different plan index, (re)compute novelty and add to database
        # none does not use features
        # When using rollout-IW, novelty is recomputed each time the node is reached
        if self.pruning in ['rollout-IW', 'rollout-IW-threshold']:
            if is_new or prev_index != plan_index:
                self.nodes_reached += 1
                node.novelty = self._novelty_db.check_and_update_novelty_table(node.data["features"],
                                                                               node.depth,
                                                                               new_node=True)
                if node.novelty == 1:
                    self.cumulative_novelty += 1
            else:
                node.novelty = self._novelty_db.check_and_update_novelty_table(node.data["features"],
                                                                               node.depth,
                                                                               new_node=False)

        # In prob-IW, we only compute the probabilistic novelty once for each node. Updates would require recomputing
        # probabilities for that each feature is in all other nodes for all nodes, which is not tractable.
        elif self.pruning == 'prob-IW':
            if is_new or prev_index != plan_index:
                self.nodes_reached += 1
                node.novelty = self._novelty_db.prob_novel(node.data["belief_state"], node.depth)
                self._novelty_db.add(node.data["belief_state"], node.depth)
                self.cumulative_novelty += node.novelty

        elif self.pruning == 'none':
            if is_new or prev_index != plan_index:
                self.nodes_reached += 1

        else:
            assert False

        # If node is the end of an episode, or reaches the max depth, finish the rollout, mark as no actions available,
        # and return 0
        if node.data["done"] or node.depth >= self.max_depth - 1:
            node.incomplete_actions = []
            return 0

        # Determine whether to continue the rollout based on the novelty.
        # none does not stop based on novelty
        # In rollout-IW, stopping due to novelty will mark the node as having no incomplete actions
        # Do not stop the rollout at the root, even if not novel (no features)
        if self.pruning in ['rollout-IW', 'rollout-IW-threshold']:
            if node.novelty > 1 and node.depth != 0:
                node.incomplete_actions = []
                return 0
        if self.pruning == 'prob-IW':
            r = random.random()
            if r > node.novelty and node.depth != 0:
                return 0

        # Select an action to perform
        action = action_f(node)

        # If we haven't sampled this action before, make a new node
        if is_new or action not in node.children:
            successor = successor_f(node, action, plan_index)
            self.simulator_calls += 1
            if self.simulator_calls >= self.simulator_limit:
                return successor.data["r"]
            q_from_successor = self.rollout(successor, True, plan_index, action_f, successor_f, update_index_f)
        else:
            successor = node.children[action]
            q_from_successor = self.rollout(successor, False, plan_index, action_f, successor_f, update_index_f)

        # Update reward distributions
        q = successor.data["r"] + self.discount * q_from_successor

        # If the successor has no incomplete actions, remove it from the list of incomplete actions
        if not successor.incomplete_actions:
            # This should never not be true
            if action in node.incomplete_actions:
                node.incomplete_actions.remove(action)

        node.update_q(action, q)

        return q
