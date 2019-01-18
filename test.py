from MarioGym import MarioGym
import matplotlib.pyplot as plt

# Get the environment and extract the number of actions.
env = MarioGym(headless=True)

restart = False
plt.ion()

counter = 0
while True:
    counter += 1
    if restart:
        env.reset()
    obs = env.level_to_supersimple_numpy()
    nummy = env.level_to_numpy()
    if nummy.shape != (0, 80) and counter % 5 == 0:
        print(nummy.shape)

        plt.matshow(nummy)
        plt.pause(1)
        plt.close()
    print(obs)
    if -90 < obs[0] < 0 and obs[2] > 0:
        obs, reward, restart, info = env.step(4)
    elif 0 < obs[0] < 90 and obs[2] < 0:
        obs, reward, restart, info = env.step(3)
    elif obs[5] < 0:
        obs, reward, restart, info = env.step(1)
    else:
        obs, reward, restart, info = env.step(0)

