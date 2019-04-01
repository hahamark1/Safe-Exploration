from MarioGym import MarioGym
import matplotlib.pyplot as plt

# Get the environment and extract the number of actions.
env = MarioGym(headless=True, level_name='Level-5-coins.json', no_coins=5)

restart = False
plt.ion()

map_length = 25
counter = 0
while True:
    counter += 1
    if restart:
        env.reset()
    obs = env.level_to_supersimple_numpy()
    nummy = env.level_to_numpy()

    obs, reward, restart, info = env.step(1)

    if env.mario.rect.x > 33*map_length:
        env.count_entities()
        if env.no_coins == 0:
            break

        env.reset_clean(env.mario.rect.y, counter)
    # if -90 < obs[0] < 0 and obs[2] > 0:
    #     obs, reward, restart, info = env.step(4)
    # elif 0 < obs[0] < 90 and obs[2] < 0:
    #     obs, reward, restart, info = env.step(3)
    # elif obs[5] < 0:
    #     obs, reward, restart, info = env.step(1)
    # else:
    #     obs, reward, restart, info = env.step(0)

