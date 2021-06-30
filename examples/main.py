import oatomobile
import oatomobile.envs
import oatomobile.baselines.rulebased

# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01")

# Makes an initial observation.
observation = environment.reset()
done = False

agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)

while not done:
  # Selects a random action.
  action = environment.action_space.sample()
  action = agent.act(observation)
  observation, reward, done, info = environment.step(action)

  # Renders interactive display.
  environment.render(mode="human")

# Book-keeping: closes
environment.close()
