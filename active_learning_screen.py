# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os
import sys
sys.path.insert(0, './VAE-IW')
import torch
from torch.utils.tensorboard import SummaryWriter
from screen import Screen
from train_encoder import make_model
from dataset import PixelDataSet, create_data_loaders
import numpy as np
from train_encoder import train_epoch, test_epoch
import random
from train_encoder import loss_function

default_loss = {
    "train_loss":      0,
    "train_kld_loss":  0,
    "train_recon_loss":0,
    "test_loss":       0,
    "test_kld_loss":   0,
    "test_recon_loss": 0,
}


class ActiveLearningScreen(Screen):
    """This class is responsible for deriving features from game screens. It is also responsible for adding data to
    screen datasets and training VAEs as appropriate.

    zdim: Size of z dimension of encoding tensor.

    xydim: Size of x/y dimensions of encoding tensor. This parameter is not currently respected.

    out_dir: Name of directory used to save screens. Will be created if it does not exist.

    feature_size: Size of feature space.

    temperature: Temperature to use in the Binary-Concrete distribution for VAE training.

    loss: One of 'BCE', 'BCE_low_prior', or 'MSE'. The loss function to use for VAE training.

    sigma: Standard deviation of normal distribution output by the decoder.

    beta: KL-loss coefficient when training beta-VAE.

    pruning: One of 'none', 'rollout-IW', or 'prob-IW'. Determines how features are used during search.

    features: One of 'B-PROST', 'VAE', or 'online-VAE'. Determines how features are computed from screens.

    zero_novel: True if a feature being zero should be treated as novel the first time it is seen. If False, only
    features equal to one are novel.

    screen_selection: One of 'entropy', 'elbo', 'loss', 'rec'. Determines how screens are added to the dataset."""

    def __init__(self, zdim, xydim, datasetsize, out_dir, feature_size,
                 max_temperature=5.0, min_temperature=0.5,
                 loss="BCE", sigma=0.1, beta=1.0, pruning='prob-IW', features='online-VAE',
                 zero_novel=False, screen_selection='entropy', screen_per_episode = 500, start_episode=0,
                 threshold=0.9,
                 screen_dataset_limit=15000, **_):
        self.zdim = zdim
        self.xydim = xydim
        self.datasetsize = datasetsize
        self.feature_size = feature_size
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.zero_novel = zero_novel
        self.screen_per_episode = screen_per_episode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(os.path.join(out_dir, 'runs'))
        self.writer_epoch = 100 * start_episode
        self.writer_entropy_count = 500 * start_episode
        self.writer_hamming_count = 0
        self.episode = start_episode

        self.loss = loss
        if loss == "MSE":
            self.normalize = True
        else:
            self.normalize = False
        self.sigma = sigma
        self.beta = beta
        self.dataset = PixelDataSet(dir=out_dir, normalize=self.normalize, limit=screen_dataset_limit)
        self.model = None

        self.pruning = pruning
        self.features = features
        self.screen_selection = screen_selection
        self.threshold = threshold

        # Nodes that will be added to the database on the next loop
        self._queued_nodes = []


    # Although the name is GetFeatures, it actually *sets* values to node.data["features"].
    # Bad student code in VAE-IW!
    def GetFeatures(self, env, new_node, prev_node=None):
        if self.pruning == 'none':
            # do nothing because we don't learn screens.
            return

        # handle zero_novel case where being 0 is also counted as a feature
        def compute_categorical_features(one_features):
            # example:
            # 0,1,2,3 feature_size = 4
            # 0,1
            #     2,3
            #     6,7
            # list all features.
            all_features = np.arange(0, self.feature_size)
            # Find the set difference of two arrays.
            zero_features = np.setdiff1d(all_features, one_features, assume_unique=True)
            # offset the indices.
            zero_features = zero_features+self.feature_size
            return np.append(one_features, zero_features)


        # If using B-PROST features, use the computation routine from the VAE-IW repository.
        # In B-PROST, the data stored in node.data["features"] is a list of indices.
        if self.features == 'B-PROST':
            self.Bprost(env, new_node, prev_node)
            if self.zero_novel:
                new_node.data["features"] = compute_categorical_features(new_node.data["features"])
            return

        # In VAEs, first compute the belief state, which is a 1D vector.
        elif self.features in ['VAE','online-VAE']:
            # compute features
            if self.model is not None:
                with torch.no_grad():
                    orig = new_node.data["obs"]
                    # convert to a batched form with a single channel, shape [1,1,xydim,xydim]
                    orig = torch.tensor(orig[None, None, :, :]/255, dtype=torch.float32, device=self.device)
                    if self.dataset.normalize:
                        orig = (orig - self.dataset.means)/self.dataset.stds
                    enc = self.model.encode(orig)
                    enc = torch.sigmoid(enc.squeeze(0).detach())
                    belief_state = np.asarray(enc.cpu().flatten())
                    del orig
                    del enc

            else:
                # model is not made yet
                belief_state = np.full((self.zdim * self.xydim**2), 0.5)
            new_node.data["belief_state"] = belief_state
            # note: even in rollout-IW, belief state could be used if combined with active learning.
        else:
            assert False

        # when using prob-IW, do nothing. prob-IW uses a belief state (probabilities) directly, rather than features.
        if self.pruning == 'prob-IW':
            return


        # We sample values when using 'rollout-IW'.
        # The data stored in node.data["features"] is the same as in B-PROST: indices.
        # note : this is different from VAE-IW, which uses a threshold = 0.9, which skews the probability.
        # See GetFeaturesFromModel in VAE-IW/screen.py
        if self.pruning == 'rollout-IW':
            rand = np.random.uniform(size=belief_state.shape)
            sample = (rand <= belief_state)
            # one_features contain indices of features that a state has.
            one_features = np.where(sample)[0]
            if self.zero_novel:
                new_node.data["features"] = compute_categorical_features(one_features)
            else:
                new_node.data["features"] = one_features
            return

        # This pruning mode is identical to VAE-IW, which uses a threshold lambda = 0.9 to compute novelty features.
        # See GetFeaturesFromModel in VAE-IW/screen.py
        if self.pruning == 'rollout-IW-threshold':
            sample = (belief_state > self.threshold)
            # one_features contain indices of features that a state has.
            one_features = np.where(sample)[0]
            if self.zero_novel:
                new_node.data["features"] = compute_categorical_features(one_features)
            else:
                new_node.data["features"] = one_features
            return

        assert False


    def queue_screens(self,nodes):
        if self.pruning == 'none':
            # do nothing because we don't learn screens.
            return
        self._queued_nodes.extend(nodes)

    def store_all_queued_screens(self):
        if self.pruning == 'none':
            # do nothing because we don't learn screens.
            return
        count = 0
        for node in self._queued_nodes:
            added = self.dataset.update_set(node.data["obs"])
            if added:
                count += 1
        self._queued_nodes = []
        return count

    def store_selected_queued_screens(self):
        if self.pruning == 'none':
            # do nothing because we don't learn screens.
            return

        if self.screen_selection == 'entropy':
            # Pick the nodes with highest entropy to add
            scores = map(lambda node: entropy(node.data["belief_state"]), self._queued_nodes)
        elif self.screen_selection == 'loss':
            # Pick the nodes with the highest loss to add. Affected by the current value of beta.
            scores = map(lambda node: self.compute_loss(node.data["obs"],beta=self.beta), self._queued_nodes)
        elif self.screen_selection == 'rec':
            # Pick the nodes with the highest reconstruction loss to add.
            scores = map(lambda node: self.compute_loss(node.data["obs"],beta=0), self._queued_nodes)
        elif self.screen_selection == 'elbo':
            # Pick the nodes with the highest elbo to add
            scores = map(lambda node: self.compute_loss(node.data["obs"],beta=1), self._queued_nodes)
        elif self.screen_selection == 'random':
            # Pick the nodes randomly
            scores = np.random.random(size=len(self._queued_nodes))
        else:
            assert False

        # sorted with reverse=True returns highest to lowest
        sorted_nodes = sorted(zip(self._queued_nodes,scores), key=lambda pair: pair[1], reverse=True)

        # Add up to 500 elements
        count = 0
        for node, score in sorted_nodes:
            added = self.dataset.update_set(node.data["obs"])
            if added:
                self.writer.add_scalar('NewScreens/NewScore', score, self.writer_entropy_count)
                self.writer_entropy_count += 1
                count += 1
            if count >= self.screen_per_episode:
                break

        self.writer.add_scalar('NewScreens/MinScore', sorted_nodes[-1][1], self.writer_entropy_count)

        self._queued_nodes = []
        torch.cuda.empty_cache()
        return count

    def retrain_model(self, epochs):
        # If using none, don't train, and return 0
        if self.pruning == 'none':
            return default_loss

        self.ensure_model()
        train_loader, test_loader, train_set, test_set = create_data_loaders(self.dataset, 64)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        for i in range(1, epochs+1):
            # note: train_epoch calls model.train() and sets it to training mode
            train_loss, train_kld_loss, train_reconstruction_loss = train_epoch(self.model, optimizer, train_loader, i,
                                                                                min_temperature=self.min_temperature,
                                                                                max_temperature=self.max_temperature,
                                                                                loss_fn=self.loss,
                                                                                sigma=self.sigma,
                                                                                beta=self.beta,
                                                                                writer=self.writer,
                                                                                writer_epoch=self.writer_epoch)
            self.writer_epoch += 1
        # note: test_epoch calls model.eval() and sets it to evaluation mode
        test_loss, test_kld_loss, test_reconstruction_loss = test_epoch(self.model, test_loader, 0,
                                                                        min_temperature=self.min_temperature,
                                                                        max_temperature=self.max_temperature,
                                                                        loss_fn=self.loss,
                                                                        sigma=self.sigma,
                                                                        beta=self.beta,
                                                                        writer=self.writer,
                                                                        writer_epoch=self.writer_epoch,
                                                                        episode=self.episode)
        self.episode += 1

        return {
            "train_loss":      train_loss,
            "train_kld_loss":  train_kld_loss,
            "train_recon_loss":train_reconstruction_loss,
            "test_loss":       test_loss,
            "test_kld_loss":   test_kld_loss,
            "test_recon_loss": test_reconstruction_loss,
        }

    def compute_loss(self, obs, beta):
        """Compute the loss for a single observation."""
        if self.model is not None:
            with torch.no_grad():
                orig = torch.tensor(obs[None, None, :, :]/255, dtype=torch.float32, device=self.device)
                if self.dataset.normalize:
                    orig = (orig - self.dataset.means)/self.dataset.stds

                recon_batch, qz, z = self.model(orig, self.min_temperature, False)
                loss, _, _, _ = loss_function(recon_batch, orig, qz, z, 1, choose_loss=self.loss,
                                              sigma=self.sigma, beta=beta)

                del orig
                return loss

        # If there is no model, return a constant loss
        else:
            return 0.0


    def make_model(self):
        self.model = make_model(zdim=self.zdim, image_training_size=self.datasetsize, image_channels=1, loss=self.loss)
    def ensure_model(self):
        if self.model is None:
            self.make_model()
    def load_model(self,path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.ensure_model()
            self.model.load_state_dict(state_dict)
        except FileNotFoundError as e:
            print(e)
    def save_model(self,path,overwrite=False):
        self.ensure_model()

        if (not overwrite) and os.path.exists(path):
            path2 = path+"_"
            print(f"the weight file {path} already exists! saving to an alternative name {path2} to avoid overwrite")
            self.save_model(path2,overwrite=False)
            return

        torch.save(self.model.state_dict(), path)



# bs : belief state
def entropy(bs):
    assert bs.ndim == 1
    # Clip between 1e-6 and 1-1e-6 for numerical stability
    p = np.clip(bs, 1e-6, 1-1e-6)
    not_p = 1 - p
    return - np.sum(p * np.log(p) + not_p * np.log(not_p))



