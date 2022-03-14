# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import numpy as np


class BeliefStateDepthDataBase:

    def __init__(self, width=1, zdim=20, zero_novel=False, max_depth=100):
        assert width == 1       # do not allow higher width now

        self._width = width
        self._zdim = zdim
        self._max_depth = max_depth
        self._zero_novel = zero_novel

        # Table containing a probability that a categorical feature is novel in a particular depth.
        # [F, 2^width, D]
        # F : number of atoms
        # 2^width = 2 : one for "true", another for "false"
        # D : maximum depth.
        self._vector_probs = np.ones((max_depth, 2, zdim))

    # bs: belief state is a 1D vector.
    def prob_novel(self, bs, depth):
        # Compute probabilities that an index is novel using cached probabilities the feature is present at depth or below.

        # important NOTE:
        # the index 0 represents a feature j is first achieved,
        # the index 1 represents a feture \not j is first achieved.
        if self._zero_novel:
            p_ind_novel =  self._vector_probs[depth, 0, :] * bs     # bs*(1-bs)*(1-bs)*(1-bs)...*(1-bs).
            p_ind_novel += self._vector_probs[depth, 1, :] * (1-bs) # (1-bs)*bs*bs*bs*...*bs.
        else:
            p_ind_novel = self._vector_probs[depth, 0, :] * bs # bs*(1-bs)*(1-bs)*(1-bs)...*(1-bs).

        # Compute probabilities that any feature is novel.
        return 1 - np.prod(1 - p_ind_novel)

    def add(self, bs, depth):

        if depth >= self._max_depth:
            # Amortized linear time array extension.
            # Since it allocates O(2^n) array on every O(2^n) steps, the amortized space is O(1)
            current = self._vector_probs
            self._vector_probs = np.ones((self._max_depth*2, 2, self._zdim))
            self._vector_probs[:self._max_depth, :, :] = current
            self._vector_probs[self._max_depth:, :, :] = self._vector_probs[self._max_depth-1, :, :]
            self._max_depth *= 2

        self._vector_probs[depth:, 0:1, :] *= (1 - bs).reshape([1,1,-1])
        self._vector_probs[depth:, 1:2, :] *= bs.reshape([1,1,-1])
