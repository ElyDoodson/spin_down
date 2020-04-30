import pickle
import datetime
# from datetime import date, time
from time import time
from gmm_methods import *


# Importing data
data_path = "C:/dev/spin_down/all_cluster_data.pickle"
all_clusters = get_data(data_path, remove_old=True)
# draw_3d_plot(all_clusters)

width_and_steps = [(0.1, 0.03), (0.15, 0.04), (0.2, 0.04),
                   (0.1, 0.03), (0.2, 0.02), (0.1, 0.02)]
# cluster_title = "ngc6811"
training_data_dict = {key: {} for key in all_clusters.keys()}

fig = plt.figure(1, figsize=(16, 9))
for i, ((bin_width, bin_step), cluster_title) in enumerate(zip(width_and_steps, all_clusters.keys())):
    # Transform data to shape [2,]
    X = transform_data(cluster_title, all_clusters)
    ########################
    # Creates bin boundaries based on X's mass values(index [:,0])
# bin_width, bin_step = 0.1, 0.02
    kernal_bins = kernal_bin_bounds(X, bin_width=bin_width, bin_step=bin_step)

    # Transforms boundaries from [1,n] -> [2,n/2]
    X_bins = pd.IntervalIndex.from_arrays(*kernal_bins.T)

    # Binned Data
    X_kernaled = kernal_binner(X, X_bins)

    # Data passed to predict proba on
    grid_x = np.linspace(0, 50, 100)

    final_curve_list = []
    for bin_n, bin_x in enumerate(X_kernaled):
        gauss = GaussianMixture(3, n_init=20)
        summed_curve = gaussian_probability(
            bin_x, grid_x, gauss)
        final_curve_list.append(summed_curve)

    final_curve_list = np.array(final_curve_list)

    bin_centres = np.array([np.average((one, two))
                            for (one, two) in kernal_bins])

    # Plot creation
    ax = plt.subplot(2, 3, i+1)
    ax.set(xlim=(1.4, 0.0), ylim=(0, 55), title="{}, Age = {}Myrs".format(
        cluster_title, all_clusters[cluster_title]["age"]/1e6))
    ax.set(ylabel="Period(days)" if i == 1 else None,
           xlabel="Mass ($M_\odot$)" if i == 5 else None)
    contour_mesh_x, contour_mesh_y = np.meshgrid(
        bin_centres, grid_x)
    ax.contourf(contour_mesh_x, contour_mesh_y,
                final_curve_list.T, 100, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], **{"s": 2, "alpha": 0.3, "c": "black"})

    training_data_dict[cluster_title].update(
        {"grid_mass": np.concatenate(contour_mesh_x)})
    training_data_dict[cluster_title].update(
        {"grid_period": np.concatenate(contour_mesh_y)})
    training_data_dict[cluster_title].update(
        {"grid_proba": np.concatenate(final_curve_list.T)})
plt.show()

pickle_file = "C:/dev/spin_down/nn_training_file.pickle"
with open(pickle_file, "wb") as file:
    pickle.dump(training_data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


# save_path = "C:/dev/spin_down/final_report/report_images/"
# now = datetime.datetime.now()
# plt.savefig(save_path +
#             "{}_bw{:.2f}_bs{:.2f}_{}.png".format(cluster_title, bin_width, bin_step, now.strftime("%d-%b-%y")))

# concat_mesh_x, concat_mesh_y, concat_curves = np.concatenate(
#     contour_mesh_x), np.concatenate(contour_mesh_y), np.concatenate(final_curve_list)
