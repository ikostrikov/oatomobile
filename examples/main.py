import oatomobile
import oatomobile.envs
import oatomobile.baselines.rulebased
from action_dict_to_array import ActionDictToArray

# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01")
environment = ActionDictToArray(environment)

# Makes an initial observation.
observation = environment.reset()
done = False

# agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)

while not done:
  # Selects a random action.
  action = environment.action_space.sample()

  action[0] = 0.0
  action[2] = 1.0

  # action = agent.act(observation)
  observation, reward, done, info = environment.step(action)

  # Renders interactive display.
  # environment.render(mode="human")

# Book-keeping: closes
environment.close()
