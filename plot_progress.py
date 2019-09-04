import matplotlib.pyplot as plt
import numpy as np
import os
import json

PATH = "saved_models/2019-09-02 17-51-26"

def load_metrics(path):
    epoch_list = []
    loss_list = []
    accuracy_list = []

    path = os.path.join(path, "checkpoints")
    for fn in os.listdir(path):
        if not fn.endswith(".json"):
            continue

        with open(os.path.join(path, fn)) as f:
            d = json.load(f)
            epoch_list.append(d["epoch"])
            loss_list.append(d["training_real_loss"])
            accuracy_list.append(d["training_accuracy"])

    epoch_list = np.array(epoch_list)
    loss_list = np.array(loss_list)
    accuracy_list = np.array(accuracy_list)
    sorted_indices = np.argsort(epoch_list)
    epoch_list = epoch_list[sorted_indices]
    loss_list = loss_list[sorted_indices]
    accuracy_list = accuracy_list[sorted_indices]

    return epoch_list, loss_list, accuracy_list
    

epoch_list, loss_list, accuracy_list = load_metrics(PATH)
plt.plot(epoch_list, loss_list)
plt.plot(epoch_list, accuracy_list)
plt.show()
