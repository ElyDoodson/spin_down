import numpy as np
import pandas as pd
from matplotlib import style, rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture

from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.exceptions import NotFittedError

style.use("ggplot")
rcParams['figure.figsize'] = 11, 5


def get_data(path, merge_clusters=True, remove_young=True, remove_old=True, remove_sparse=True):
    assert type(path) == str
    all_clusters = pd.DataFrame.from_dict(
        pd.read_pickle(path, compression=None))

    if remove_young:
        del all_clusters["usco"]

    if remove_sparse:
        del all_clusters["h_per"]
        del all_clusters["alpha_per"]
        del all_clusters["hyades"]
        del all_clusters["ngc2547"]

    if remove_old:
        del all_clusters["ngc6819"]
        del all_clusters["m67"]

    cluster_merger(all_clusters, "pleiades_m50", "pleiades", "m50")
    cluster_merger(all_clusters, "m35_ngc2516", "m35", "ngc2516")
    cluster_merger(all_clusters, "ngc2301_m34", "ngc2301", "m34")
    all_clusters = all_clusters.sort_values(by=["age"], axis=1)
    return all_clusters


def draw_3d_plot(data_frame):
    # 3D plot of all clusters remaining
    all_clusters = data_frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=20., azim=-35)
    ax.set_zlim((0, 30))
    for key in all_clusters.keys():
        below_1_3 = all_clusters[key]["Mass"] < 1.3
        age, period, mass = all_clusters[key]["age"], all_clusters[
            key]["Per"][below_1_3], all_clusters[key]["Mass"][below_1_3]
        ax.scatter(mass, age, period, s=3, alpha=0.2)
    plt.show()


def transform_data(cluster_name, data_frame):
    """
    transforms two elements of a data frame into a 2d array of [ele1,ele2]
    """
    mass = data_frame[cluster_name].copy()["Mass"][:, np.newaxis]
    per = data_frame[cluster_name].copy()["Per"][:, np.newaxis]

    X = np.concatenate((mass, per), axis=1)
    if cluster_name == "m35_ngc2516":
        X = remove_above(X, 25., 1.25)
    else:
        X = remove_above(X, None, 1.25)

    return X


# x_lower, x_upper, y_lower, y_upper, x_num, y_num):
def get_uniform_grid(linspace_x, linspace_y, transform=True):

    # uniform_xdata = np.linspace(x_lower, x_upper, x_num)
    # uniform_ydata = np.linspace(y_lower, y_upper, y_num)
    uniform_xdata = linspace_x
    uniform_ydata = linspace_y
    mesh_x, mesh_y = np.meshgrid(uniform_xdata, uniform_ydata)
    uniform_grid = np.concatenate((np.concatenate(mesh_x)[:, np.newaxis],
                                   np.concatenate(mesh_y)[:, np.newaxis]), axis=1)
    return (uniform_grid) if transform else (mesh_x, mesh_y)


def get_binned_grid(bin_brackets, num_x=20, num_y=25, y_value=25):
    all_grid_bin = []
    for x_lower, x_upper in zip(bin_brackets, bin_brackets[1:]):
        grid_bin = get_uniform_grid(x_lower, x_upper, 0, y_value, num_x, num_y)
        all_grid_bin.append(grid_bin)

    return np.array(all_grid_bin)


def data_binner(number_of_bins=2, method="same_width"):
    """Generates function to bin data

    Paramerters
    -----------
    number_of_bins : int, optional
        Number of Bins. Defaults to 2.
    method : {"same_width", "same_length"}
        Selecting pandas method for binning data. 
        "same_width" yields pandas.cut(), "same_length" yields pandas.qcut()

    Returns
    ------
    data_binner : function
        Contains definded parameters to pass data and returns binned data.
    """
    # Data type assertions
    assert method == "same_width" or "same_length"
    # assert type(number_of_bins) == int

    def func(data, axis=0, return_brackets=False):
        """Bins data along a specified axis

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a single data point.

        axis : int, optional
            Value refering to data slice by which to bin data. Defaults to 1.
            axis <= n_features

        return_brackets : bool, optional
            When True, returns the bin brackets with the bins

        Returns
        -------
        _ : array-like, shape (number_of_bins, )
            List of binned data in specified axis
        """

        if axis >= data.shape[0]:
            raise ValueError("Axis out of range of data features. \
            Data features = {}, selected axis = {}".format(data.shape[0], axis))

        if method == "same_width":
            categoricals, bin_brackets = pd.cut(
                data[:, axis], number_of_bins, retbins=True)

        if method == "same_length":
            categoricals, bin_brackets = pd.qcut(
                data[:, axis], number_of_bins, retbins=True)

        list_of_bins = [[] for _ in range(len(bin_brackets)-1)]

        for bin_index, datum in zip(categoricals.codes, data):
            list_of_bins[bin_index].append(np.array(datum))

        array_of_bins = np.array([np.array(bin) for bin in list_of_bins])
        if return_brackets == True:
            return array_of_bins, bin_brackets
        else:
            return array_of_bins

    return func


def kernal_binner(data_to_bin, bins):

    bin_indexes = []
    for x_y in data_to_bin:
        bin_mapping = [x_y[0] in _bin for _bin in bins]
        idx = [i for i, _ in enumerate(bin_mapping) if _]
        bin_indexes.append(idx)

    binned_kernal = [[] for _ in range(len(bins))]

    for current_bin, datum in zip(bin_indexes, data_to_bin):
        for c_bin in current_bin:
            binned_kernal[c_bin].append(list(datum))

    return np.array([np.array(i) for i in binned_kernal])


def gm_clusterer(data, gm_model):
    """ Groups the data according to the fitted Gaussian Mixture model provided

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The data which you want to bin using the model passed
    gm_model : obj
        A fitted gaussian mixture model(gmm).

    Returns
    -------
    _ : array-like, shape (number_of_clusters, )
        A list of the data in their clusters predicted by the gmm.
    """

    check_is_fitted(gm_model)
    number_of_clusters = len(gm_model.means_)

    cluster_list = [[] for _ in range(number_of_clusters)]
    check_array(data)

    reshaped_data = data[:, 1][:, np.newaxis] if len(data.shape) == 1 else data
    predicted_cluster = gm_model.predict(reshaped_data)

    for pred_cluster, datum in zip(predicted_cluster, data):
        cluster_list[pred_cluster].append(np.array(datum))

    length_array = np.array([len(item) for item in cluster_list])
    if (length_array == 0).any():
        print("Index of bins with 0 elements: ", *
              ["{},".format(idx) for idx, item in enumerate(length_array) if item == 0])
    return np.array([np.array(cluster) for cluster in cluster_list])


def gaussian_probability(real_data, grid_data, gauss_model):

    real_data_ = real_data[:, 1][:, np.newaxis]
    grid_data_ = grid_data[:, np.newaxis]

    gmm = gauss_model.fit(real_data_)
    cluster_probabilities = gmm.predict_proba(grid_data_)
    data_clusters = gm_clusterer(real_data_, gmm)

    bin_probas = []
    for index, cluster in enumerate(data_clusters):
        cluster_prob = cluster_probabilities[:, index]
        summed_proba = np.sum(cluster_prob)
        normalised_prob = cluster_prob / \
            summed_proba if summed_proba != 0.0 else cluster_prob / 1.0
        length_normalised_proba = normalised_prob * \
            (len(cluster)/len(grid_data))
        bin_probas.append(length_normalised_proba)

    return np.sum(bin_probas, axis=0)


def cluster_merger(df, new_name, cluster1, cluster2):
    """
    Changes df in place, adding "new_name" by combining and then deleting cluster1/2 
    """
    period1, period2 = df[cluster1]["Per"], df[cluster2]["Per"]
    mass1, mass2 = df[cluster1]["Mass"], df[cluster2]["Mass"]

    avg_age = np.average((df[cluster1]["age"], df[cluster2]["age"]))

    df[new_name] = [avg_age, np.append(
        period1, period2), np.append(mass1, mass2), []]
    del df[cluster1]
    del df[cluster2]


def remove_above(data_list, period=None, mass=None):
    if period != None:
        assert type(period) == type(1.)
        data_list_mask = data_list[:, 1] > period
        data_list = data_list[~data_list_mask]

    if mass != None:
        assert type(mass) == type(1.)
        data_list_mask = data_list[:, 0] > mass
        data_list = data_list[~data_list_mask]

    return data_list


def reverse_scatter(data, model=None, plt_args={}, gaussian_centres=False, highlighting=False, return_figax=False, figure=None):
    if figure == None:
        fig, ax = plt.subplots(1, figsize=(10, 5))
        # fig,ax = plt.subplots(1)
    else:
        assert type(figure) == tuple
        fig, ax = figure

    # Standarising plots
    ax.set(ylim=(-1, 27), xlim=(1.3, 0.1), ylabel="Period", xlabel="Mass")

    if model == None:
        ax.scatter(data[:, 0], data[:, 1], **plt_args)

    if model != None:
        assert type(model) == GaussianMixture
        # cluster_index = model.predict(data[:,1][:,np.newaxis])
        # # print("cluster_index" + str(cluster_index))
        # cluster_list = cluster_from_index(data, cluster_index)
        # cluster_lengths.sort()
        cluster_list = gm_clusterer(data, model)

        cluster_lengths = np.array([len(c) for c in cluster_list])

        # for i, cluster in enumerate(cluster_list):
        #     plot_centre = False
        #     if len(cluster) >= cluster_lengths[-2]:
        #         alpha = 0.9 if highlighting else 1
        #         plot_centre = True
        #     else:
        #         alpha = 0.3 if highlighting else 1
        #     ax.scatter(cluster[:,0], cluster[:,1], alpha = alpha, c = "grey" if (len(cluster) == max(cluster_lengths)) & highlighting else None, **plt_args)
        #     ax.scatter(np.average(cluster[:,0]), model.means_[i] ,color = "black", marker = "x", s = 50) if (gaussian_centres) & (plot_centre)  else None

        for i, cluster in enumerate(cluster_list):
            try:
                ax.scatter(cluster[:, 0], cluster[:, 1], **plt_args)
                ax.scatter(np.average(cluster[:, 0]), model.means_[
                           i], color="", marker="x", s=50) if gaussian_centres else None
            except IndexError:
                print("Not plotting index {}, no elements found".format(i))

    if return_figax == True:
        return fig, ax
    elif return_figax == False:
        fig.show()
    else:
        return -1


def kernal_bin_bounds(data, bin_width, bin_step, starting_value=None):
    max_value, min_value = max(data[:, 0]), min(data[:, 0])
    if starting_value == None:
        starting_value = min_value

    if bin_step > bin_width:
        print("WARNING: Some data will be unbinned, bin step is larger than the bin size.")

    near_bin_edge = starting_value
    far_bin_edge = starting_value + bin_width

    bin_list = [np.array([near_bin_edge, far_bin_edge])]
    while far_bin_edge < max_value:
        near_bin_edge += bin_step
        far_bin_edge += bin_step
        bin_list.append(np.array([near_bin_edge, far_bin_edge]))
    print("Returning {} bins".format(len(bin_list)))
    return np.array(bin_list)
