import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from A2 import mapFeature

def plot_decision(lr, d_list, X1, X2, y):
    """Plot decision boundary for multiple degrees and return cross validation errors."""
    grid_size = 200
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    xx, yy = xx.ravel(), yy.ravel()

    cross_errors = []
    plt.figure(f"C = {lr.C}", figsize=(12, 7))
    for i in d_list:
        Xe = mapFeature(X1, X2, i, False)
        lr.fit(Xe, y)
        test_grid_e = mapFeature(xx, yy, i, False)
        pred_grid = lr.predict(test_grid_e)
        y_pred = lr.predict(Xe)
        cross_errors.append(np.sum(y != cross_val_predict(lr, Xe, y)))
        plt.subplot(3, 3, i)
        plt.gca().set_title(f"D: {i}, Training errors = {np.sum(y!=y_pred)}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.imshow(pred_grid.reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
        plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")
    plt.tight_layout()
    return cross_errors

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/microchips.csv", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    X1, X2 = X[:, 0], X[:, 1]

    d_list = range(1, 10)
    lr = LogisticRegression(C=10000., solver="lbfgs", max_iter=1000)
    ce_10000 = plot_decision(lr, d_list, X1, X2, y)

    lr2 = LogisticRegression(C=1., solver="lbfgs", max_iter=1000)
    ce_1 = plot_decision(lr2, d_list, X1, X2, y)

    plt.figure("Degree d vs #errors")
    plt.plot(d_list, ce_10000, label=f"C = {lr.C}")
    plt.plot(d_list, ce_1, label=f"C = {lr2.C}")
    plt.xlabel("Degree d")
    plt.ylabel("# errors")
    plt.legend()


if __name__ == "__main__":
    main()
    plt.show()