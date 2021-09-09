import logging
import os
import json

import numpy as np
import ray
from gym.spaces import Box, Discrete
from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer, APPOTrainer
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray import tune
from ray.tune import register_env
from ray.tune.logger import pretty_print
from ray.tune.integration.wandb import WandbLoggerCallback

import rlgym
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import *
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.rllib_utils import RLLibEnv
import wandb
from reward import TimeLeftEventReward


MAX_EP_SECS = 10
DEFAULT_TICK_SKIP = 8
PHYSICS_TICKS_PER_SECOND = 120
MAX_EP_STEPS = int(round(MAX_EP_SECS * PHYSICS_TICKS_PER_SECOND / DEFAULT_TICK_SKIP))
ENV_CONFIG = {
    "self_play": True,
    "team_size": 1,
    "game_speed": 100,
    "obs_builder": AdvancedObs(),
    "reward_fn": CombinedReward(
        (
            TimeLeftEventReward(goal=1, concede=-1, shot=0.01, save=0.01),
            # LiuDistanceBallToGoalReward(),
            VelocityBallToGoalReward(),
            # RewardIfBehindBall(TouchBallReward()),
            # LiuDistancePlayerToBallReward(),
            # VelocityPlayerToBallReward(),
            # AlignBallGoal(),
            # RewardIfBehindBall(FaceBallReward()),
            # VelocityReward(),
            ConstantReward(),
        ),
        (1, 1, -0.01)
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


if __name__ == '__main__':
    ray.init(address='auto', _redis_password='5241590000000000', logging_level=logging.DEBUG)


    def create_env(env_config):
        return RLLibEnv(rlgym.make(**ENV_CONFIG))


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
        callbacks=[
            # WandbLoggerCallback(
            #     project="ray-rlbot",
            #     log_config=True,
            #     config={
            #         "wandb": {
            #             "settings": {
            #                 "start_method": "spawn"
            #             },
            #         },
            #         "max_ep_secs": MAX_EP_SECS,
            #         "env_config": json.loads(json.dumps(ENV_CONFIG, default=json_dumper)),
            #     }
            # )
        ]
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