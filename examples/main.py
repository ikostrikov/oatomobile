import oatomobile
import oatomobile.envs
import oatomobile.baselines.rulebased
from action_dict_to_array import ActionDictToArray
from jaxrl.wrappers import TakeKey
from tqdm import tqdm
import imageio


# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01") #, sensors=['front_camera_rgb'])
environment = ActionDictToArray(environment)
# environment = TakeKey(environment, 'front_camera_rgb')

# Makes an initial observation.
observation = environment.reset()
done = False

# agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)

for i in range(10):
  print("Pre Reset")
  observation = environment.reset()
  print("Post Reset")
  done = False

  frames = []
  for _ in tqdm(range(200)):
    # Selects a random action.
    action = environment.action_space.sample()

    action[0] = 0.0
    action[2] = 1.0

    # action = agent.act(observation)
    observation, reward, done, info = environment.step(action)

    # Renders interactive display.
    frame = environment.render(mode="rgb_array")
    frames.append(frame)

  imageio.mimsave(f'videos/{i}.mp4', frames, fps=20)
  frames = []

# Book-keeping: closes
environment.close()
