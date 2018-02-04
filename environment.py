import gym
import feature_extraction

class GymEnvironment():

  def __init__(self, env_id, frames_to_images, debug):
    self.env_id = env_id
    self.gym = gym.make(env_id)
    self.obs = None
    self.terminal = None

    self.ip = feature_extraction.ImageProcessor(env_id, frames_to_images, debug)

    # Define actions for games (gym-0.9.4 and ALE 0.5.1)
    if env_id == "Pong-v0":
      self.action_space = [1, 2, 3] # [NONE, UP, DOWN]
    elif env_id == "Breakout-v0":
      self.action_space = [1, 2, 3] # [FIRE, RIGHT, LEFT]
    else:
      self.action_space = [i for i in range(0, self.gym.action_space.n)] # 9 discrete actions are available

  def numActions(self):
    assert isinstance(self.gym.action_space, gym.spaces.Discrete)
    return len(self.action_space)

  def featureVectorSize(self):
      return self.ip.get_feature_vector_size()

  def restart(self):
    self.obs = self.gym.reset()
    self.terminal = False

  def act(self, action):
    self.obs, reward, self.terminal, _ = self.gym.step(self.action_space[action])
    return reward, self.terminal

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal

  def render(self):
    self.gym.render()

  def getFeatures(self):
    assert self.obs is not None
    return self.ip.pipeline(self.obs)

