import numpy as np
from rlgym.utils import common_values, math
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.terminal_conditions import TerminalCondition


class CustomObsBuilderBluePerspective(ObsBuilder):
  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
    obs = []
    
    #If this observation is being built for a player on the orange team, we need to invert all the physics data we use.
    inverted = player.team_num == common_values.ORANGE_TEAM
    
    if inverted:
      obs += state.inverted_ball.serialize()
    else:
      obs += state.ball.serialize()
      
    for player in state.players:
      if inverted:
        obs += player.inverted_car_data.serialize()
      else:
        obs += player.car_data.serialize()
    
    return np.asarray(obs, dtype=np.float32)


class SpeedReward(RewardFunction):
  def reset(self, initial_state):
    pass

  def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    linear_velocity = player.car_data.linear_velocity
    reward = math.vecmag(linear_velocity)

    return reward
    
  def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
    return 0


class CustomTerminalCondition(TerminalCondition):
  def reset(self, initial_state: GameState):
    pass

  def is_terminal(self, current_state: GameState) -> bool:
    return current_state.last_touch != -1