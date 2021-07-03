import oatomobile
import oatomobile.envs
import oatomobile.baselines.rulebased
from action_dict_to_array import ActionDictToArray
from tqdm import tqdm

# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01",
      sensors=["bird_view_camera_rgb"])
environment = ActionDictToArray(environment)
import ipdb; ipdb.set_trace()

# Makes an initial observation.
observation = environment.reset()
done = False

# agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)

for _ in range(10):
  print("Pre Reset")
  observation = environment.reset()
  print("Post Reset")
  done = False
  for _ in tqdm(range(100)):
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
