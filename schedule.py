#!/usr/bin/env python3

# This file is a part of atari-active-learning project.
# Copyright (c) 2022 Benjamin Ayton (aytonb@mit.edu), Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

# set the log level to info
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import os
import os.path
import luigi
import pandas as pd
import itertools
import stacktrace
import glob
import json
import numpy as np
import multiprocessing as mp
import uuid

class AbstractExperiment(luigi.Task):
    #### meta parameters
    dry = luigi.BoolParameter(default=False,
                              significant=False,
                              description="dry-run. Just print the command, without running it.")
    cluster = luigi.BoolParameter(default=False,
                                  significant=False,
                                  description="when true, use CCC cluster's internal scheduling system called spjb.")
    pjobs = luigi.IntParameter(default=0,
                               significant=False,
                               description="*job scheduler specific parameter* Specify the maximum number of jobs that are allowed to run in the same project. When zero, no limit.")
    root = luigi.Parameter(default="results")

    # Don't automatically retry. Usually failed due to a timeout or disk space issue that retries won't fix.
    retry_count = 0
    pass


class SingleActiveRolloutExperiment(AbstractExperiment):
    #### command line parameters
    seed = luigi.IntParameter(default=2021)
    env = luigi.Parameter()
    loss = luigi.Parameter(default='BCE_std_prior')
    beta = luigi.FloatParameter(default=0.0001)
    pruning = luigi.Parameter(default="rollout-IW-threshold")
    bandit = luigi.Parameter(default='TTTS')
    features = luigi.Parameter(default='online-VAE')
    screen_selection = luigi.Parameter(default='elbo')
    epochs = luigi.IntParameter(default=100)
    zero_novel = luigi.BoolParameter(default=False)
    max_temperature = luigi.FloatParameter(default=5.0)
    reuse = luigi.BoolParameter(default=False)
    budget_per_action = luigi.IntParameter(default=100)
    max_episode_length_during_training = luigi.IntParameter(default=200)

    test = luigi.BoolParameter(default=False)
    proj = luigi.Parameter(default="al",
                           significant=False,
                           description="project name to be displayed in the LSF job scheduler.")
    mem = luigi.IntParameter(default=16,significant=False)
    maxmem = luigi.IntParameter(default=512,significant=False)
    uuid4 = luigi.Parameter(significant=False)
    gpu = luigi.BoolParameter(True,significant=False)
    time = luigi.IntParameter(default=12,significant=False)

    def requires(self):
        return None

    @property
    def name(self):
        return "_".join(map(str,[
            self.env, self.loss, self.beta,
            self.pruning, self.bandit, self.features,
            self.screen_selection,
            self.max_temperature,
            self.epochs,
            self.reuse,
            self.budget_per_action,
            self.max_episode_length_during_training,
        ]))

    def output(self):
        return luigi.LocalTarget(os.path.join(self.root,self.name,str(self.seed),"done.json"))

    def weights(self):
        return os.path.join(self.root,self.name,str(self.seed),"weights")

    def cmd(self):
        cmd = []
        if self.cluster:
            cmd += ["jbsub", "-mem", f"{self.mem}g"]
            cmd += ["-wait"]
            if self.test: # queue selection. on test, run quick
                self.time = 1
            cmd += ["-q", f"x86_{self.time}h"]
            if self.features == 'B-PROST':
                # Run only with CPU with B-PROST
                self.gpu = False
            if self.gpu:
                # Run with CPU + v100/a100 GPU (blocking cccxc444 and cccxc506 due to gpu ECC memory error )
                cmd += ["-cores", "1+1", "-require", "'(v100||a100)'"]
            else:
                cmd += ["-cores", "1"]
            cmd += ["-proj", self.proj]
            cmd += ["-name", self.uuid4]
            cmd += ["-pjobs", self.pjobs]

        cmd += ["python", "-u", "main.py"]
        cmd += [self.env]
        cmd += [os.path.join(self.root,self.name,str(self.seed))]
        cmd += ["--seed", self.seed]
        cmd += ["--loss", self.loss]
        cmd += ["--beta", self.beta]
        cmd += ["--pruning", self.pruning]
        cmd += ["--bandit", self.bandit]
        cmd += ["--features", self.features]
        cmd += ["--screen-selection", self.screen_selection]
        cmd += ["--epochs", self.epochs]
        cmd += ["--weights", os.path.join(self.root,self.name,str(self.seed),"weights")]
        cmd += ["--zero-novel", self.zero_novel]
        cmd += ["--max-temperature", self.max_temperature]
        cmd += ["--budget", self.budget_per_action]
        cmd += ["--max-episode-length-during-training", self.max_episode_length_during_training]
        if self.reuse:
            cmd += ["--reuse-model"]

        if self.test: # run quick
            # in online VAE, this will be bootstrap + about 2 episodes.
            cmd += ["--budget", 10]
            cmd += ["--total-training-budget", 500]
            cmd += ["--max-episode-length", 20]
            cmd += ["--screen-per-episode", 10]
            cmd += ["--screen-dataset-limit", 25]
            cmd += ["--max-rollout-length", 5]
            cmd += ["--evals", 1]

        return " ".join(map(str,cmd))

    def run(self):
        if self.cluster:
            # TERM_OWNER: job killed by owner.
            # Exited with exit code 130.
            # TERM_RUNLIMIT: job killed after reaching LSF run time limit.
            # Exited with exit code 140.
            # TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.
            # Exited with exit code 137.
            while True:
                if self.dry:
                    print("dry run:",self.cmd())
                    return
                print(self.cmd())
                result = os.system(self.cmd())
                exitstatus, signal = result >> 8, result % 256
                if exitstatus == 0:
                    print(f"Job finished")
                    break
                elif exitstatus == 3:
                    print(f"Training phase is finished; submit another job for evaluation using CPU")
                    self.gpu = False
                elif exitstatus == 137:
                    # memory limit; double the memory, resubmit and continue
                    self.mem *= 2
                    if self.mem > self.maxmem:
                        print(f"max memory limit {self.maxmem}g is reached... not attempting further retry.")
                        break
                    print(f"memory limit; double the memory to {self.mem}g, resubmit and continue.")
                else:
                    print(f"unexpected error code {exitstatus}... not attempting a further retry.")
                    break
        else:
            if self.dry:
                print("dry run:",self.cmd())
                return
            print(self.cmd())
            os.system(self.cmd())
        return


# Full set of catesian products of all parameters worth investigating.
# However, this is actually too expensive, and some parameters do not make sense
# seed x envs x 3 x 3 x 2 x 3 x 6 x 3 x 2 = seed x envs x 1944
class ExhaustiveExperiment(AbstractExperiment):
    seeds_start = luigi.IntParameter(default=0)
    seeds = luigi.IntParameter(default=5)
    envs = ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
            'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout',
            'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
            'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
            'IceHockey', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster',
            'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pong',
            'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
            'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
            'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon']
    # note: sort in order of importance.
    losses = ['BCE_zerosup', 'BCE_low_prior', 'BCE_std_prior',]
    betas  = [0.0001, 1.0, 0.0,]
    prunings = ['prob-IW', 'rollout-IW', 'none','rollout-IW-threshold']
    bandits = ['TTTS', 'uniform', 'UCT', 'max', 'TS', 'SH',]
    features = ['online-VAE','B-PROST','VAE',]
    screen_selections = ['elbo', 'entropy', 'rec', 'loss', 'random']
    zero_novels = [True,False]

    def configurations(self):
        configs = {
            "seed": range(self.seeds_start,self.seeds),
            "env": self.envs,
            "loss": self.losses,
            "beta": self.betas,
            "pruning": self.prunings,
            "bandit": self.bandits,
            "features": self.features,
            "screen_selection": self.screen_selections,
            "zero_novel": self.zero_novels,
        }
        for config in itertools.product(*(configs.values())):
            yield dict(zip(configs.keys(), config))


    def requires(self):
        reqs = {}
        import itertools
        for config in self.configurations():
            reqs[tuple(config.items())] = \
                SingleActiveRolloutExperiment(root=self.root,
                                              cluster=self.cluster,
                                              dry=self.dry,
                                              uuid4="j"+str(uuid.uuid4()), # job name should not start with digits
                                              pjobs=self.pjobs,
                                              **config)
        return reqs

    def output(self):
        return None

    def run(self):
        pass


class BprostExperiment(ExhaustiveExperiment):
    def configurations(self):
        keys = ["proj", "seed", "env", "bandit", "features"]

        configs = [
            ('uniform', 'B-PROST'),
        ]
        for (config, seed, env, ) in itertools.product(configs, range(self.seeds_start,self.seeds), self.envs, ):
            yield dict(zip(keys, ("albprost", seed, env, *config)))


class VAEIWBaselineExperiment(ExhaustiveExperiment):
    def configurations(self):
        keys = ["proj", "seed", "env", "bandit", "features", "max_temperature"]
        # with / without annealing
        configs = [
            ('uniform', 'VAE', 5.0),
            ('uniform', 'VAE', 0.5),
        ]
        for (config, seed, env, ) in itertools.product(configs, range(self.seeds_start,self.seeds), self.envs, ):
            yield dict(zip(keys, ("alvae", seed, env, *config)))


class VAEIWBanditExperiment(ExhaustiveExperiment):
    def configurations(self):
        keys = ["proj", "seed", "env", "bandit", "features"]
        configs = [
            # Compare the effect of different multiarmed bandit methods.
            ('uniform', 'VAE'),
            ('TTTS', 'VAE'),
            ('UCT', 'VAE'),
            ('max', 'VAE'),
        ]
        for (config, seed, env, ) in itertools.product(configs, range(self.seeds_start,self.seeds), self.envs, ):
            yield dict(zip(keys, ("alvaebandit", seed, env, *config)))


class OnlineVAEExperiment(ExhaustiveExperiment):
    """Evaluate Active Olive and Passive Olive."""
    def configurations(self):
        keys = ["proj", "seed", "env", "screen_selection"]
        # compare online-VAE with active learning (elbo) and passive learning (random)
        screen_selections = ["elbo", "random"] # , "entropy", "rec", "loss",
        for (ss, seed, env, ) in itertools.product(screen_selections, range(self.seeds_start,self.seeds), self.envs, ):
            yield dict(zip(keys, ("alonline", seed, env, ss)))


# unused in the paper
class OnlineVAEBudgetExperiment(ExhaustiveExperiment):
    """Compare runtime planning budgets and episode length cutoff."""
    def configurations(self):
        keys    = ["proj", "seed", "env", "budget_per_action", "max_episode_length_during_training"]
        budgets = [100, 200]
        lengths = [200, 18000]
        # compare longer budget with all other configurations being equal
        for (b, l, seed, env, ) in itertools.product(budgets, lengths, range(self.seeds_start,self.seeds), self.envs, ):
            yield dict(zip(keys, ("alonlinebudget", seed, env, b, l)))


# unused in the paper
class OnlineVAEReuseExperiment(OnlineVAEExperiment):
    """See the effect of inheriting the weights from the previous iteration"""
    def configurations(self):
        keys = ["proj", "seed", "env", "reuse"]
        reuse = [True, False]
        # compare online-VAE with and without reuse, all other configurations being equal
        for (r, seed, env, ) in itertools.product(reuse, range(self.seeds_start,self.seeds), self.envs, ):
            yield dict(zip(keys, ("alreuse", seed, env, r)))



class AllExperiment(ExhaustiveExperiment):
    def requires(self):
        return [
            self.clone(BprostExperiment),
            self.clone(VAEIWBaselineExperiment),
            self.clone(VAEIWBanditExperiment),
            self.clone(OnlineVAEExperiment),
            # not used in the paper.
            # self.clone(OnlineVAEBudgetExperiment),
            # self.clone(OnlineVAEReuseExperiment),
        ]



################################################################
#### tests

class TestExperiment(ExhaustiveExperiment):
    seeds = luigi.IntParameter(default=1)
    envs = ['Alien',]
    def configurations(self):
        keys = ["proj", "test", "seed", "env", "loss", "pruning", "bandit", "features", "screen_selection", "epochs"]

        configs = [
            (self.losses[0], self.prunings[0], self.bandits[0], self.features[0], self.screen_selections[0], 2),
            *[(x,              self.prunings[0], self.bandits[0], self.features[0], self.screen_selections[0], 2) for x in self.losses[1:]],
            *[(self.losses[0], x,                self.bandits[0], self.features[0], self.screen_selections[0], 2) for x in self.prunings[1:]],
            *[(self.losses[0], self.prunings[0], x,               self.features[0], self.screen_selections[0], 2) for x in self.bandits[1:]],
            *[(self.losses[0], self.prunings[0], self.bandits[0], x,                self.screen_selections[0], 2) for x in self.features[1:]],
            *[(self.losses[0], self.prunings[0], self.bandits[0], self.features[0], x,                         2) for x in self.screen_selections[1:]],
        ]
        for (seed, env, config) in itertools.product(range(self.seeds_start,self.seeds), self.envs, configs):
            yield dict(zip(keys, ("altest", True, seed, env, *config)))


# Test 100 randomly selected command line parameters.
class RandomTestExperiment(ExhaustiveExperiment):
    seeds = luigi.IntParameter(default=1)
    tests = luigi.IntParameter(default=100)
    def configurations(self):
        import random
        reservoir = []
        for i, config in enumerate(super().configurations()):
            if len(reservoir) < self.tests:
                reservoir.append(config)
            else:
                # when i=100, it is 101-th element.
                j = random.randrange(i+1)
                if j < self.tests:
                    reservoir[j] = config

        for config in reservoir:
            yield {"proj":"alrandomtest", "test":True, **config}



################################################################
#### evaluation

import aggregate_pandas as ap

class AbstractEvaluation(luigi.Task):
    def evalname(self,base=None):
        return self.__class__.__name__


class VAEIWBanditComparison(AbstractEvaluation):
    def requires(self):
        return self.clone(VAEIWBaselineExperiment)

    def run(self):
        ap.dump_results(
            self.evalname(),
            "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_*_100_False_100_200/*")


class VAEIWOfflineComparison(AbstractEvaluation):
    """Compare RIW, VAEIW, and VAEIW+ann (annealing enabled), VAEIW+ann+TTTS"""
    def requires(self):
        return self.clone(VAEIWBanditExperiment)

    def run(self):
        ap.dump_results(
            self.evalname(),
            ["results/*_BCE_std_prior_0.0001_rollout-IW_uniform_B-PROST_elbo_5.0_100_False_100_200/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_0.5_100_False_100_200/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_5.0_100_False_100_200/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_VAE_elbo_5.0_100_False_100_200/*",
             ],
            ["pruning","loss"])


class BudgetComparison(AbstractEvaluation):
    def requires(self):
        return self.clone(OnlineVAEEpisodeLengthExperiment)

    def run(self):
        # limiting the episode length
        ap.dump_results_pairwise(
            self.evalname(),
            ["results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_200/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_200_200/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_18000/*",
             "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_200_18000/*"])


class AblationStudy(AbstractEvaluation):
    def requires(self):
        return self.clone(AllExperiment)

    def run(self):
        # RIW,VAEIW,VAEIW+ann,VAEIW+ann+TTTS,PassiveOlive,ActiveOlive,PortfolioOlive
        ap.dump_results(
            self.evalname(),
            [
                "results/*_BCE_std_prior_0.0001_rollout-IW_uniform_B-PROST_elbo_5.0_100_False_100_200/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_0.5_100_False_100_200/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_uniform_VAE_elbo_5.0_100_False_100_200/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_VAE_elbo_5.0_100_False_100_200/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_random_5.0_100_False_100_200/*",
                "results/*_BCE_std_prior_0.0001_rollout-IW-threshold_TTTS_online-VAE_elbo_5.0_100_False_100_200/*",
            ],
            ["pruning","loss"],portfolio=True)



class AllAnalysis(AbstractEvaluation):
    def requires(self):
        return [
            self.clone(VAEIWBanditComparison),
            self.clone(VAEIWOfflineComparison),
            # self.clone(BudgetComparison),
            self.clone(AblationStudy),
        ]




if __name__ == '__main__':
    try:
        luigi.run()
    except:
        stacktrace.format()
