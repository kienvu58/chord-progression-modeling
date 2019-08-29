import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def softmax(a, t):
    a = a / t
    a_exp = np.exp(a)
    return a_exp/a_exp.sum()

# weight = torch.load("data/targets/target_1-1-1-1-1-1.th")
# token_weight = weight[2][2:30]
# print(token_weight)
sim_vec = np.array([8, 4, 2])
sim_vec = sim_vec / sim_vec.sum()

def plot(sim_vec, name, title):

    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=True,
        labeltop=False,
        labelright=False,
        labelbottom=True,
    )
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    plt.ylim((0, 1))
    x_pos = np.arange(len(sim_vec))
    bar = plt.bar(x_pos, sim_vec, width=0.9)

    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % height, ha='center', va='bottom')

    plt.xticks(np.arange(3), [r"F$\sharp$maj", r"G$\flat$maj", r"Cmaj"])
    plt.title(title, y=1)

    plt.savefig("{}.svg".format(name))
    plt.clf()

plot(sim_vec, "sim_vec", "Similarity vector")

t = 1
decay = 3
for i in range(6):
    sim_target = softmax(sim_vec, t)
    # plot(sim_target, str(i), r"Similarity target at $T_{} = {:.2f}$".format(i, t))
    plot(sim_target, str(i), r"$T_{} = {:.2f}$".format(i, t))
    t = t / (1 + decay * t)



