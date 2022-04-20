from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import A2

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/microchips.csv", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    X1, X2 = X[:, 0], X[:, 1]
    
    grid_size = 1000
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    xx, yy = xx.ravel(), yy.ravel()

    plt.figure("Original chip data", figsize=(12, 7))
    plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")


    #[5.16939737   3.2475016    4.16623174 -12.02686332  -7.53125756
    # -11.8224845 ]
    # 0.34808750706317243

    ############################################################################
    ## Exercise 4: Task 2                                                     ##
    ############################################################################
    Xe = np.c_[np.ones((X.shape[0], 1)), X1, X2, X1**2, X1*X2, X2**2]
    alpha, N = 5, 10000
    beta, costs = A2.logistic_gradient_descent(Xe, y, alpha, N, with_costs=True)
    print("Hyperparameters:")
    print(f"  α = {alpha}")
    print(f"  N = {N}")

    plt.figure("Cost over iterations and decision boundary", figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.gca().set_title("J(β) over iterations")
    plt.gca().set_box_aspect(1)
    plt.xlabel("Iterations (N)")
    plt.ylabel("Cost (J)")
    plt.plot(range(N), costs)


    test_grid_e = np.c_[np.ones(xx.shape), xx, yy, xx**2, xx*yy, yy**2]
    p_grid = A2.sigmoid(np.dot(test_grid_e, beta))
    pp = np.round(A2.sigmoid(np.dot(Xe, beta)))

    plt.subplot(1, 2, 2)
    plt.gca().set_title(f"Training errors = {np.sum(y!=pp)}")
    plt.gca().set_box_aspect(1)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.imshow((p_grid>0.5).reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
    plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")


    ############################################################################
    ## Exercise 4: Task 4                                                     ##
    ############################################################################
    Xe_d5 = A2.mapFeature(X1, X2, 5)
    # alpha, N = 17, 5000000
    alpha, N = 7, 200000
    beta, costs = A2.logistic_gradient_descent(Xe_d5, y, alpha, N, with_costs=True)
    print("Hyperparameters:")
    print(f"  α = {alpha}")
    print(f"  N = {N}")

    plt.figure("Cost over iterations and decision boundary (d=5)", figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.gca().set_title("J(β) over iterations")
    plt.gca().set_box_aspect(1)
    plt.xlabel("Iterations (N)")
    plt.ylabel("Cost (J)")
    plt.plot(range(N), costs)


    test_grid_e = A2.mapFeature(xx, yy, 5)
    p_grid = A2.sigmoid(np.dot(test_grid_e, beta))
    pp = np.round(A2.sigmoid(np.dot(Xe_d5, beta)))

    plt.subplot(1, 2, 2)
    plt.gca().set_title(f"Training errors = {np.sum(y!=pp)}")
    plt.gca().set_box_aspect(1)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.imshow((p_grid>0.5).reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaffaa"]))
    plt.scatter(X1, X2, c=y, cmap=ListedColormap(["red", "green"]), marker=".")

    # [  2.2039123   -4.12359785   1.57163643   1.12219579 -16.43147534
    #    7.52570018  24.06668607  35.53962242  41.08454913   2.42573466
    #  -14.75468415  14.44390454 -17.05868867  16.02102529 -34.49203441
    #  -22.61932889 -39.75929752 -26.94068763 -64.37612239 -66.96994072
    #   16.08540372]


if __name__ == "__main__":
    main()
    plt.show()