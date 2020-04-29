from time import sleep
from gmm_methods import *

data_path = "C:/dev/spin_down/all_cluster_data.pickle"
all_clusters = get_data(data_path)

# draw_3d_plot(all_clusters)

X = transform_data("pleiades_m50", all_clusters)

# # Non-Overlapping bin creation
# num_of_bins = 8
# binner = data_binner(num_of_bins, "same_length")
# binned_X, bin_brackets = binner(X, axis=0, return_brackets=True)
# # Each bin has N number of points
# binned_X_grid = get_binned_grid(bin_brackets)

# Overlapping bin creation
kernal_bins = kernal_bin_bounds(X, bin_width=0.1, bin_step=0.02)
X_bins = pd.IntervalIndex.from_arrays(*kernal_bins.T)


X_kernaled = kernal_binner(X, X_bins)


# uniform_grid = get_uniform_grid(
#     min(X[:, 0]), max(X[:, 0]), 0, 25, x_num=100, y_num=30)
# # Bin has varying number of points
# X_kernaled_grid = kernal_binner(uniform_grid, X_bins)

grid_x = np.linspace(0, 40, 100)

# Multi-attribute figure
fig = plt.figure(1, figsize=(16, 9))
# Non-Overlapping bins

# ax = plt.subplot(2, 2, 1)
# ax.set(xlim=(1.3, 0.2), ylim=(0, 45), title="Non-Overlapping Bins")
# final_curve_list = []

# for bin_n, bin_x in enumerate(binned_X):
#     gauss = GaussianMixture(5, n_init=5)

#     summed_curve = gaussian_probability(
#         bin_x, grid_x, gauss, multi_dim=False)
#     final_curve_list.append(summed_curve)

# final_curve_list = np.array(final_curve_list)

# bin_centres = np.array([np.average((one, two))
#                         for (one, two) in zip(bin_brackets, bin_brackets[1:])])

# contour_mesh_x, contour_mesh_y = np.meshgrid(
#     bin_centres, grid_x)
# ax.contourf(contour_mesh_x, contour_mesh_y,
#             final_curve_list.T, 30, cmap='inferno')

# ax.scatter(X[:, 0], X[:, 1], **{"s": 1, "alpha": 0.3, "c": "white"})
# # # Overalpping bins
# # fig, ax = plt.subplots(1, figsize=(11, 4))

ax = plt.subplot(111)
ax.set(xlim=(1.3, 0.2), ylim=(0, 30))
# ax.text(0.1, 0, "1D, Each bin has 1D Prob Dist", rotation=270, size=14)
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
            final_curve_list.T, 30, cmap='inferno')
ax.scatter(X[:, 0], X[:, 1], **{"s": 1, "alpha": 0.3, "c": "white"})
plt.show()
# # reverse_scatter(X, figure=(fig, ax), plt_args={
# #                 "s": 1, "alpha": 0.3, "c": "white"})

# # plt.gca().set(xlabel="Mass", ylabel="Period")
# # plt.show()


# # Testing with grid of probabilities

# # fig, ax = plt.subplots(1, figsize=(10, 5))

# ax = plt.subplot(2, 2, 3)
# ax.set(xlim=(1.3, 0.2), ylim=(0, 30))
# final_curve_list = []
# for bin_ in binned_X:

#     gmm = GaussianMixture(5, n_init=10)
#     grid_of_probs = gaussian_probability(
#         bin_, uniform_grid, gmm, multi_dim=True)

#     final_curve_list.append(grid_of_probs)

# summed_grid = np.sum(final_curve_list, axis=0)

# ax.contourf(uniform_grid[:, 0].reshape(30, 100), uniform_grid[:, 1].reshape(30, 100),
#             summed_grid.reshape(30, 100), 30, cmap='inferno')
# ax.scatter(X[:, 0], X[:, 1], **{"s": 1, "alpha": 0.3, "c": "white"})

# ax = plt.subplot(2, 2, 4)
# ax.set(xlim=(1.3, 0.2), ylim=(0, 30))

# ax.text(0.1, 0, "2D, Each bin has 2D grid Prob Dist", rotation=270, size=14)
# final_curve_list = []
# for bin_ in X_kernaled:

#     gmm = GaussianMixture(5, n_init=10)
#     grid_of_probs = gaussian_probability(
#         bin_, uniform_grid, gmm, multi_dim=True)

#     final_curve_list.append(grid_of_probs)

# summed_grid = np.sum(final_curve_list, axis=0)

# ax.contourf(uniform_grid[:, 0].reshape(30, 100), uniform_grid[:, 1].reshape(30, 100),
#             summed_grid.reshape(30, 100), 30, cmap='inferno')

# ax.scatter(X[:, 0], X[:, 1], **{"s": 1, "alpha": 0.3, "c": "white"})
# plt.show()
