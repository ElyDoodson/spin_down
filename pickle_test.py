import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn import Linear, Module, ModuleList, MSELoss, ReLU, Tanh
from torch.optim import Adam
from tqdm import tqdm

from gmm_methods import get_data, get_uniform_grid, transform_data


class NN(Module):
    def __init__(self, num_layers, num_nodes, activation):
        super(NN, self).__init__()

        # Specify list of layer sizes
        sizes = [3] + [num_nodes] * num_layers + [1]
        in_sizes, out_sizes = sizes[:-1], sizes[1:]

        # Construct linear layers
        self.linears = ModuleList()
        for n_in, n_out in zip(in_sizes, out_sizes):
            self.linears.append(Linear(n_in, n_out))

        # Specify activation function
        self.activation = activation

    def forward(self, x):

        for l in self.linears[:-1]:
            x = self.activation(l(x))
        x = self.linears[-1](x)

        return x


def extract_features_labels(data_frame):
    _dict = {key: {} for key in data_frame.keys()}
    for cluster_nm in data_frame.keys():
        cluster = data_frame[cluster_nm]
        age = cluster["age"]
        concat_list = np.array([cluster["grid_mass"], cluster["grid_period"]])

        transposed_list = np.transpose(concat_list)

        feature_list = np.array([np.append(array, age/1e6)
                                 for array in transposed_list])
        feature_label_list = np.concatenate(
            (feature_list, cluster["grid_proba"][:, np.newaxis]), axis=1)
        _dict[cluster_nm].update({"data": feature_label_list})

    return _dict


def shuffle_data(input_dict):

    shuffled_data = []
    for cluster_nm in input_dict:
        cluster = input_dict[cluster_nm]["data"]

        np.random.shuffle(cluster)
        shuffled_data.append(cluster)

    shuffled_data = np.concatenate(shuffled_data)
    np.random.shuffle(shuffled_data)

    return shuffled_data


def train_test_split(data, train_test_ratio=0.2):
    training_set, test_set = [], []

    test_size = int(len(data) * train_test_ratio)

    training_set.append(data[:-test_size])
    test_set.append(data[-test_size:])
    training_set = np.asarray(training_set)
    test_set = np.asarray(test_set)
    return training_set[0], test_set[0]


def format_data(input_, train_test_ratio=0.2):
    one = extract_features_labels(input_)

    two = shuffle_data(one)
    # print(two)
    three = train_test_split(two, train_test_ratio)
    return three


pickle_file = "C:/dev/spin_down/nn_training_file.pickle"
training_dataframe = pd.DataFrame.from_dict(
    pd.read_pickle(pickle_file, compression=None))

training_set, testing_set = format_data(training_dataframe, 0.1)

x_train, y_train = training_set[:, 0:3], training_set[:, -1][:, np.newaxis]
x_test, y_test = testing_set[:, 0:3], testing_set[:, -1][:, np.newaxis]

x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
x_test, y_test = torch.tensor(x_test).float(), torch.tensor(y_test).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if True:
    net = NN(3, 20, Tanh()).to(device)
else:
    net = NN(3, 20, Tanh()).to(device)
    net.load_state_dict(torch.load("C:/dev/spin_down/network_params.pt"))
    net.to(device)

optimiser = Adam(net.parameters(), lr=0.010)
loss_function = MSELoss()

times, epochs, losses, val_losses = [], [], [], []


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimiser.step()
    return loss


def test():
    with torch.no_grad():
        val_loss = fwd_pass(x_test.to(device), y_test.to(device))
    return val_loss


def train():
    BATCH_SIZE = 128
    EPOCHS = 4050
    plt.figure(figsize=(16, 9))

    for epoch in tqdm(range(EPOCHS)):
        train_loss_list = []
        val_loss_list = []

        permutation = torch.randperm(x_train.size()[0])

        for i in range(0, len(x_train), BATCH_SIZE):
            # Shuffles the data each pass
            indices = permutation[i:i+BATCH_SIZE]
            batch_X, batch_y = x_train[indices].to(
                device), y_train[indices].to(device)

            # Loss for training data + network train
            loss = fwd_pass(batch_X, batch_y, train=True)
            train_loss_list.append(float(loss * len(batch_X)))

            # loss for test data
            val_loss = test()
            val_loss_list.append(float(val_loss * len(batch_X)))

        # times.append(float(round(time.time(),3)))
        epochs.append(float(epoch))

        losses.append(np.mean(train_loss_list))
        val_losses.append(np.mean(val_loss_list))

    plt.plot(epochs, losses, label="Train")
    plt.plot(epochs, val_losses, label="Test")
    plt.legend()
    torch.save(net.state_dict(), "C:/dev/spin_down/network_params.pt")


train()
plt.gca().set(xlabel="Epochs", ylabel="Log Loss", yscale="log")
plt.show()


# data_path = "C:/dev/spin_down/all_cluster_data.pickle"
# all_clusters = get_data(data_path, remove_old=True)
# cluster_name = "m37"
# cluster_age = all_clusters[cluster_name]["age"]
# X = transform_data(cluster_name, all_clusters)
# X = torch.tensor([np.append(mass_per, cluster_age/1e6)
#                   for mass_per in X]).float().to(device)
xdata, ydata = np.linspace(0.1, 1.3, 30), np.linspace(0, 35, 100)
X = get_uniform_grid(xdata, ydata)
X = torch.Tensor([np.append(mass_per, 500)
                  for mass_per in X]).float().to(device)
with torch.no_grad():
    plt.contourf(*get_uniform_grid(xdata, ydata,
                                   transform=False), net(X).reshape(100, 30).cpu(), 100)
    plt.show()
