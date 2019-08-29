import numpy as np
from allennlp.data.vocabulary import Vocabulary
import torch
import os
import matplotlib.pyplot as plt

def save_confusion_matrix_figure(path, name, matrix, xlabels, ylabels):
    plt.figure()

    plt.matshow(matrix, cmap="viridis")
    # plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90, fontsize=5)
    # plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=5)

    # for i in range(matrix.shape[0]):
    #     row_denom = np.max(matrix[i])
    #     for j in range(matrix.shape[1]):
    #         value = matrix[i][j]
    #         percentage = value / row_denom * 100
    #         color = "black" if percentage > 50 else "white"
    #         plt.text(
    #             j,
    #             i,
    #             "{:0.2f}".format(value),
    #             ha="center",
    #             va="center",
    #             color=color,
    #             fontsize=5,
    #         )

    if not os.path.isdir(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, "{}.pdf".format(name)))


def plot_weight(weight_path): 
    weight = torch.load(weight_path)
    weight = weight.numpy()
    vocab = Vocabulary().from_files("data/vocabulary")
    xlabels = ylabels = list(vocab.get_token_to_index_vocabulary())
    if not os.path.isdir("figures"):
        os.makedirs("figures")
    fn = os.path.basename(weight_path)
    fn = os.path.splitext(fn)[0]
    save_confusion_matrix_figure("figures", fn, weight, xlabels, ylabels)

if __name__ == "__main__":
    plot_weight("data/targets/target_distance_2.th")