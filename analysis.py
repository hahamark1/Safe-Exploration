import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_coins_collected():
    with open('coins_collected.pickle', 'rb') as handle:
        coins_collected = pickle.load(handle)

    with open('goombas_killed.pickle', 'rb') as handle:
        goombas_killed = pickle.load(handle)

    window_width = 40

    cumsum_vec = np.cumsum(np.insert(coins_collected, 0, 0))
    ma_coins = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    cumsum_vec = np.cumsum(np.insert(goombas_killed, 0, 0))
    ma_goombas = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

    plt.plot(ma_coins)
    plt.plot(ma_goombas)
    plt.show()


plot_coins_collected()