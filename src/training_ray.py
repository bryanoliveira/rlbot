import logging

import numpy as np
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from gym.spaces import Box, Discrete
import rlgym
from rlgym.gamelaunch import LaunchPreference
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import *
from rlgym.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    GoalScoredCondition,
)
from rlgym_tools.rllib_utils import RLLibEnv
from rlgym_tools.extra_obs.general_stacking import GeneralStacker

from reward import TimeLeftEventReward, RewardIfFacingBall


MAX_EP_SECS = 20
DEFAULT_TICK_SKIP = 8
PHYSICS_TICKS_PER_SECOND = 120
MAX_EP_STEPS = int(round(MAX_EP_SECS * PHYSICS_TICKS_PER_SECOND / DEFAULT_TICK_SKIP))
ENV_CONFIG = {
    "self_play": True,
    "team_size": 1,
    "game_speed": 100,
    "spawn_opponents": False,
    "launch_preference": LaunchPreference.EPIC,
    "obs_builder": GeneralStacker(AdvancedObs(), stack_size=3),
    "reward_fn": CombinedReward(
        (
            TimeLeftEventReward(goal=1, concede=-1, shot=0.5, save=0.5),
            TouchBallReward(),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            RewardIfFacingBall(VelocityPlayerToBallReward()),
            # ConstantReward(),
            # LiuDistanceBallToGoalReward(),
            # RewardIfBehindBall(TouchBallReward()),
            # LiuDistancePlayerToBallReward(),
            # AlignBallGoal(),
            # RewardIfBehindBall(FaceBallReward()),
            # VelocityReward(),
        ),
        (1, 0.3, 0.05, 0.1),  # , -0.1
    ),
    "terminal_conditions": (TimeoutCondition(MAX_EP_STEPS), GoalScoredCondition()),
}


def json_dumper(obj):
    """
    Dumps generic objects to JSON. Useful to serialize ENV_CONFIG.
    """
    # extra hacky, but works
    try:
        return obj.toJSON()
    except:
        try:
            return obj.__dict__
        except:
            return str(obj)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "learning_agent"  # Choose 01 policy for agent_01
    else:
        return np.random.choice(
            ["learning_agent", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.85, 0.05, 0.05, 0.05],
        )[0]


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when reward is high enough
        """
        if info["result"]["episode_reward_mean"] > 30:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["learning_agent"])[
                        "learning_agent"
                    ],
                }
            )


if __name__ == "__main__":
    ray.init()

    def create_env(*_):
        """
        Instantiate the environment and wrap it in an RLLibEnv, ignoring args.
        """
        return RLLibEnv(rlgym.make(**ENV_CONFIG))

    register_env("RLGym", create_env)
    obs_space = Box(-np.inf, np.inf, (107 * 3,))
    act_space = Box(-1.0, 1.0, (8,))
    # action_space = Discrete(8,) # e.g. curiosity

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_1",
        # "SAC",
        # name="SAC_multiagent_3",
        config={
            # system settings
            "num_gpus": 1,
            # "num_cpus_for_driver": 6,
            "num_workers": 0,
            # "num_envs_per_worker": 1,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayUpdateCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "learning_agent": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["learning_agent"],
            },
            "env": "RLGym",
            # algorithm settings
            # "model": {
            #     "vf_share_layers": True,
            # },
            # "vf_loss_coeff": tune.grid_search([0.33, 0.66, 1.0, 1.33]),
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
            # "exploration_config": {
            #     "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            #     "eta": 50.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            #     "lr": 0.0003,  # Learning rate of the curiosity (ICM) module.
            #     "feature_dim": 64,  # Dimensionality of the generated feature vectors.
            #     # Setup of the feature net (used to encode observations into feature (latent) vectors).
            #     "feature_net_config": {
            #         "fcnet_hiddens": [256],
            #         "fcnet_activation": "relu",
            #     },
            #     "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            #     "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            #     "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            #     "forward_net_activation": "relu",  # Activation of the "forward" model.
            #     "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            #     # Specify, which exploration sub-type to use (usually, the algo's "default"
            #     # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            #     "sub_exploration": {
            #         "type": "StochasticSampling",
            #     }
            # }
        },
        stop={
            # "training_iteration": 10000,
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore="./ray_results/PPO_selfplay_1/PPO_RLGym_b34cc_00000_0_2021-11-09_00-25-34/checkpoint_001300/checkpoint-1300",
        # resume=True,
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
