import matplotlib.pyplot as plt


def sample_temperature(num_samples, num_epochs, T_intial, decay_rate):
    T = T_intial
    T_list = []
    for epoch in range(num_epochs):
        T *= 1 / (1 + decay_rate * epoch)
        T_list.append(T)

    step = num_epochs // num_samples
    return T_list[::step]


T_list = sample_temperature(200, 200, 0.05, 0.001)
plt.plot(T_list)
plt.show()
T_list = sample_temperature(10, 200, 0.05, 0.001)
print(T_list)