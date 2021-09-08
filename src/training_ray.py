import logging
import os

import numpy as np
import ray
from gym.spaces import Box, Discrete
from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer, APPOTrainer
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray import tune
from ray.tune import register_env
from ray.tune.logger import pretty_print

import rlgym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import *
from rlgym_tools.rllib_utils import RLLibEnv

if __name__ == '__main__':
    ray.init(address='auto', _redis_password='5241590000000000', logging_level=logging.DEBUG)


    def create_env(env_config):
        return RLLibEnv(
            rlgym.make(
                self_play=True,
                game_speed=100,
                obs_builder=AdvancedObs(),
                reward_fn=CombinedReward(
                    (
                        LiuDistanceBallToGoalReward(), # 0.01
                        VelocityBallToGoalReward(), # 0.01
                        TouchBallReward(), # 0.005
                        LiuDistancePlayerToBallReward(), # 0.0005
                        VelocityPlayerToBallReward(), # 0.0005
                        AlignBallGoal(), # 0.0001
                        FaceBallReward(), # 0.0001
                        VelocityReward(), # 0.0001
                    ), 
                    (0.01, 0.01, 0.005, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001)
                )
            )
        )


    register_env("RLGym", create_env)

    policy = PPOTorchPolicy, Box(-np.inf, np.inf, (107,)), Box(-1.0, 1.0, (8,)), {}
    # policy = SACTorchPolicy, Box(-np.inf, np.inf, (107,)), Box(-1.0, 1.0, (8,)), {}
    # policy = PPOTorchPolicy, Box(-np.inf, np.inf, (4,)), Discrete(2), {}
    
    analysis = tune.run(
        "PPO", # "SAC",
        name="PPO_multiagent_2", # name="SAC_multiagent_2",
        config={
            # system settings
            "num_gpus": 1,
            # "num_cpus_for_driver": 6,
            "num_workers": 0,
            # "num_envs_per_worker": 1,
            "log_level": "INFO",
            "framework": "torch",
            # RL setuptingsup
            "multiagent": {
                "policies": {"policy": policy},
                "policy_mapping_fn": (lambda agent_id, **kwargs: "policy"),
                "policies_to_train": ["policy"],
            },
            "env": "RLGym",
            "env_config": {
                "num_agents": 2
            },
            # algorithm settings
            "model": {
                "vf_share_layers": True,
            },
            # "lr": 5e-5,
            # "lambda": 0.95,
            # "kl_coeff": 0.2,
            # "clip_rewards": False,
            # "vf_clip_param": 10.0,
            # "entropy_coeff": 0.01,
            # "train_batch_size": 2000,
            # "sgd_minibatch_size": 500,
            # "num_sgd_iter": 10,
            # "batch_mode": "truncate_episodes",
        },
        stop={
            "training_iteration": 10000,
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/...",
        # resume=True,
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max") 
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(best_checkpoint)
    print("Done training")