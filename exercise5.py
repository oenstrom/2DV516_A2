from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from A2 import mapFeature

def plot_decision(lr, X1, X2, y):
    grid_size = 1000
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    xx, yy = xx.ravel(), yy.ravel()

    plt.figure(f"C = {lr.C}", figsize=(12, 7))
    for i in range(1, 10):
        Xe = mapFeature(X1, X2, i, False)
        lr.fit(Xe, y)
        test_grid_e = mapFeature(xx, yy, i, False)
        pred_grid = lr.predict(test_grid_e)
        y_pred = lr.predict(Xe)
        plt.subplot(3, 3, i)
        plt.gca().set_title(f"D: {i}, Training errors = {np.sum(y!=y_pred)}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.imshow(pred_grid.reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
        plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")
    plt.tight_layout()

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/microchips.csv", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    X1, X2 = X[:, 0], X[:, 1]

    # grid_size = 1000
    # x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    # y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    # xx, yy = xx.ravel(), yy.ravel()

    lr = LogisticRegression(C=10000., solver="lbfgs", max_iter=1000)
    plot_decision(lr, X1, X2, y)

    lr = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
    plot_decision(lr, X1, X2, y)

    # plt.figure("C = 10000", figsize=(12, 7))
    # for i in range(1, 10):
    #     Xe = mapFeature(X1, X2, i, False)
    #     lr.fit(Xe, y)
    #     test_grid_e = mapFeature(xx, yy, i, False)
    #     pred_grid = lr.predict(test_grid_e)
    #     y_pred = lr.predict(Xe)
    #     plt.subplot(3, 3, i)
    #     plt.gca().set_title(f"D: {i}, Training errors = {np.sum(y!=y_pred)}")
    #     plt.xlabel("Feature 1")
    #     plt.ylabel("Feature 2")
    #     plt.imshow(pred_grid.reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
    #     plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")
    # plt.tight_layout()


    # lr = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
    # plt.figure("C = 1", figsize=(12, 7))
    # for i in range(1, 10):
    #     Xe = mapFeature(X1, X2, i, False)
    #     lr.fit(Xe, y)
    #     test_grid_e = mapFeature(xx, yy, i, False)
    #     pred_grid = lr.predict(test_grid_e)
    #     y_pred = lr.predict(Xe)
    #     plt.subplot(3, 3, i)
    #     plt.gca().set_title(f"D: {i}, Training errors = {np.sum(y!=y_pred)}")
    #     plt.xlabel("Feature 1")
    #     plt.ylabel("Feature 2")
    #     plt.imshow(pred_grid.reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
    #     plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")
    # plt.tight_layout()


if __name__ == "__main__":
    main()
    plt.show()