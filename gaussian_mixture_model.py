from time import sleep
from gmm_methods import *

# Importing data
data_path = "C:/dev/spin_down/all_cluster_data.pickle"
all_clusters = get_data(data_path)

# draw_3d_plot(all_clusters)

# Transform data to shape [2,]
X = transform_data("pleiades_m50", all_clusters)

# Creates bin boundaries based on X's mass values(index [:,0])
kernal_bins = kernal_bin_bounds(X, bin_width=0.1, bin_step=0.02)

# Transforms boundaries from [1,n] -> [2,n/2]
X_bins = pd.IntervalIndex.from_arrays(*kernal_bins.T)

# Binned Data
X_kernaled = kernal_binner(X, X_bins)

# Data passed to predict proba on
grid_x = np.linspace(0, 40, 100)

# Plot creation
fig = plt.figure(1, figsize=(16, 9))
ax = plt.subplot(111)
ax.set(xlim=(1.4, 0.0), ylim=(0, 30))

final_curve_list = []
for bin_n, bin_x in enumerate(X_kernaled):

    gauss = GaussianMixture(5, n_init=20)
    summed_curve = gaussian_probability(
        bin_x, grid_x, gauss, multi_dim=False)

    # reverse_scatter(bin_x, gauss.fit(
    #     bin_x[:, 1][:, np.newaxis]), figure=(fig, ax))

    # bin_width = np.subtract(*kernal_bins[bin_n])
    # ax.plot((summed_curve * -bin_width/max(summed_curve)) +
    #         kernal_bins[bin_n][0], grid_x, c="black")
    final_curve_list.append(summed_curve)

final_curve_list = np.array(final_curve_list)

bin_centres = np.array([np.average((one, two))
                        for (one, two) in kernal_bins])

contour_mesh_x, contour_mesh_y = np.meshgrid(
    bin_centres, grid_x)
ax.contourf(contour_mesh_x, contour_mesh_y,
            final_curve_list.T, 30, cmap='Pastel1')
ax.scatter(X[:, 0], X[:, 1], **{"s": 1, "alpha": 0.3, "c": "white"})
plt.show()
