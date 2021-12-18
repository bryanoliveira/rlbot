# Here we import the Match object and our multi-instance wrapper
from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv, sb3_log_reward

# Since we can't use the normal rlgym.make() function, we need to import all the default configuration objects to give to our Match.
from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym.utils.reward_functions.common_rewards import *

# Finally, we import the SB3 implementation of PPO.
from stable_baselines3.ppo import PPO
from stable_baselines3.common import callbacks, monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from training_ray import ENV_CONFIG
from reward import TimeLeftEventReward, RewardIfFacingBall

SB_CONFIG = {
    "reward_function": sb3_log_reward.SB3CombinedLogReward(
        (
            TimeLeftEventReward(goal=1, concede=-1, shot=0.5, save=0.5),
            # TouchBallReward(),
            RewardIfTouchedLast(VelocityBallToGoalReward()),
            # RewardIfFacingBall(VelocityPlayerToBallReward()),
            # ConstantReward(),
            # LiuDistanceBallToGoalReward(),
            # RewardIfBehindBall(TouchBallReward()),
            # LiuDistancePlayerToBallReward(),
            # AlignBallGoal(),
            # RewardIfBehindBall(FaceBallReward()),
            # VelocityReward(),
        ),
        (10, 0.1),
    ),
    "terminal_conditions": ENV_CONFIG["terminal_conditions"],
    "obs_builder": ENV_CONFIG["obs_builder"],
    "state_setter": RandomState(),  # DefaultState(),
    "self_play": True,
    "team_size": 3,
    "game_speed": 200,
}
# run = wandb.init(
#     project="rlbot", entity="bryanoliveira", config=SB_CONFIG, sync_tensorboard=True
# )
class DummyRun:
    def __init__(self):
        self.id = "1_multiagent_ppo"


run = DummyRun()

# This is the function we need to provide to our SB3MultipleInstanceEnv to construct a match. Note that this function MUST return a Match object.
def get_match():
    # Here we configure our Match. If you want to use custom configuration objects, make sure to replace the default arguments here with instances of the objects you want.
    return Match(**SB_CONFIG)


# If we want to spawn new processes, we have to make sure our program starts in a proper Python entry point.
if __name__ == "__main__":
    """
    Now all we have to do is make an instance of the SB3MultipleInstanceEnv and pass it our get_match function, the number of instances we'd like to open, and how long it should wait between instances.
    This wait_time argument is important because if multiple Rocket League clients are opened in quick succession, they will cause each other to crash. The exact reason this happens is unknown to us,
    but the easiest solution is to delay for some period of time between launching clients. The amount of required delay will depend on your hardware, so make sure to change this number if your Rocket League
    clients are crashing before they fully launch.
    """
    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, num_instances=2, wait_time=20
    )
    # learner = PPO.load(
    #     "./sb3_results/rl1_model_20600000_steps",
    learner = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=2,
        tensorboard_log=f"sb3_results/{run.id}/tensorboard_logs",
    )
    learner.learn(
        100_000_000,
        callback=[
            callbacks.CheckpointCallback(
                50_000, f"sb3_results/{run.id}", name_prefix="model", verbose=2
            ),
            sb3_log_reward.SB3CombinedLogRewardCallback(
                [
                    "TimeLeftEventReward goal=1, concede=-1, shot=0.5, save=0.5",
                    "RewardIfTouchedLast VelocityBallToGoalReward",
                ]
            )
            # WandbCallback(),
        ],
    )
