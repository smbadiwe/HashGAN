# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def learn_tsne(grid, clusterer):
    Y = TSNE(n_components=3, perplexity=30.0).fit_transform(grid)
    print("Y:", Y.shape)
    # colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # n_colors = colors.shape[0]
    # print("n colors:", n_colors, "shape:", colors.shape)
    # colors = [colors[x % n_colors] for x in clusterer.labels_ if x > -1]
    fig = plt.figure()
    # ax = plt.axes()
    ax = Axes3D(fig)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=clusterer.labels_.astype(float))
    plt.show()  # savefig('name2d.png')

