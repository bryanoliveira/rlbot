import rlgym
from stable_baselines3.ppo import PPO
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition

from training_ray import ENV_CONFIG


env = rlgym.make(
    **{
        **ENV_CONFIG,
        "game_speed": 1,
        "team_size": 3,
        "terminal_conditions": (GoalScoredCondition(),),
    }
)

model = PPO.load(
    "./sb3_results/1_multiagent_ppo/model_61800000_steps.zip",
    env=env,
    verbose=1,
    policy="MlpPolicy",
)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()
