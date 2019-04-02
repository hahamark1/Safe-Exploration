from MarioGym import MarioGym
import matplotlib.pyplot as plt

# Get the environment and extract the number of actions.
env = MarioGym(headless=False, level_name='Level-5-coins.json', no_coins=5)

restart = False
plt.ion()

map_length = env.level.levelLength
print(map_length)
counter = 0
while True:
    counter += 1
    if restart:
        env.reset()
    obs = env.level_to_supersimple_numpy()
    nummy = env.level_to_numpy()

    obs, reward, restart, info = env.step(1)
    # print("\nThe current x position is {}, which is smaller than {}".format(env.mario.rect.x, 33 * env.level.levelLength))
    if env.mario.rect.x > 30.9*map_length:
        print("\nThe current x position is {}, which is larger than {}".format(env.mario.rect.x,
                                                                                33 * env.level.levelLength))

        env.count_entities()
        if env.no_coins == 0:
            break

        env.reset_clean(env.mario.rect.y)
    # if -90 < obs[0] < 0 and obs[2] > 0:
    #     obs, reward, restart, info = env.step(4)
    # elif 0 < obs[0] < 90 and obs[2] < 0:
    #     obs, reward, restart, info = env.step(3)
    # elif obs[5] < 0:
    #     obs, reward, restart, info = env.step(1)
    # else:
    #     obs, reward, restart, info = env.step(0)

