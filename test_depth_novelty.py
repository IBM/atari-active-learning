import numpy as np
from depth_novelty import BeliefStateDepthDataBase

# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation


def test_belief_state_depth_database():
    db = BeliefStateDepthDataBase(width=1, zdim=3, zero_novel=True, max_depth=3)

    bs11 = np.array([[0.8, 0.2], [0.6, 0.4], [0.2, 0.8]])

    # Prob novel at depth 0 or below = 1.0
    assert db.prob_novel(bs11, 0) == 1.0
    # Prob novel at depth 1 or below = 1.0
    assert db.prob_novel(bs11, 1) == 1.0

    db.add(bs11, 0)
    bs12 = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    db.add(bs12, 0)

    bs21 = np.array([[0.4, 0.6], [0.9, 0.1], [0.7, 0.3]])
    db.add(bs21, 1)

    bs31 = np.array([[0.8, 0.2], [0.6, 0.4], [0.8, 0.2]])
    db.add(bs31, 2)

    bsnew = np.array([[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]])
    # Prob novel at depth 0 or below = 1 - (1 - 0.7*0.9*0.8 - 0.3*0.1*0.2)*(1 - 0.3*0.2*0.6 - 0.7*0.8*0.4)
    #                                        *(1 - 0.5*0.8*0.2 - 0.5*0.2*0.8) = 0.695416
    assert abs(db.prob_novel(bsnew, 0) - (1 - (1 - 0.7*0.9*0.8 - 0.3*0.1*0.2)*(1 - 0.3*0.2*0.6 - 0.7*0.8*0.4)*\
           (1 - 0.5*0.8*0.2 - 0.5*0.2*0.8))) <= 1e-12

    # Prob novel at depth 1 or below = 1 - (1 - 0.7*0.4*0.9*0.8 - 0.3*0.6*0.1*0.2)*
    #                         (1 - 0.3*0.9*0.2*0.6 - 0.7*0.1*0.8*0.4)*(1 - 0.5*0.7*0.8*0.2 - 0.5*0.3*0.2*0.8) = 0.30885
    assert abs(db.prob_novel(bsnew, 1) - (1 - (1 - 0.7*0.4*0.9*0.8 - 0.3*0.6*0.1*0.2)*\
           (1 - 0.3*0.9*0.2*0.6 - 0.7*0.1*0.8*0.4)*(1 - 0.5*0.7*0.8*0.2 - 0.5*0.3*0.2*0.8))) <= 1e-12

    # Prob novel at depth 2 or below = 1 - (1 - 0.7*0.8*0.4*0.9*0.8 - 0.3*0.2*0.6*0.1*0.2)*
    #  (1 - 0.3*0.6*0.9*0.2*0.6 - 0.7*0.4*0.1*0.8*0.4)*(1 - 0.5*0.8*0.7*0.8*0.2 - 0.5*0.2*0.3*0.2*0.8) = 0.22618
    assert abs(db.prob_novel(bsnew, 2) - (1 - (1 - 0.7*0.8*0.4*0.9*0.8 - 0.3*0.2*0.6*0.1*0.2)*\
           (1 - 0.3*0.6*0.9*0.2*0.6 - 0.7*0.4*0.1*0.8*0.4)*(1 - 0.5*0.8*0.7*0.8*0.2 - 0.5*0.2*0.3*0.2*0.8))) <= 1e-12
