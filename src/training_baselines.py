import rlgym
from stable_baselines3 import PPO
from rlgym.utils.timeout_conditions.common_conditions import TimeoutCondition

from env import SpeedReward, CustomObsBuilderBluePerspective, CustomTerminalCondition

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 20
max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

#Make the default rlgym environment
env = rlgym.make(
    reward_fn=SpeedReward(), 
    obs_builder=CustomObsBuilderBluePerspective(), 
    terminal_conditions=[CustomTerminalCondition(), TimeoutCondition(max_steps)]
)

#Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
model.learn(total_timesteps=int(1e6))
