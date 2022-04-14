from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import A2

def grid(X1, X2, grid_size = 100):
    x_min, x_max = X1.min() - 0.1, X1.max() + 0.1
    y_min, y_max = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/admission.csv", delimiter=",")
    X   = data[:, :-1]
    y   = data[:, -1]
    Xn  = A2.normalize(X, X)
    Xne = A2.extend_matrix(Xn)


    plt.scatter(Xn[:, 0], Xn[:, 1], c=y, marker=".", cmap=ListedColormap(["red", "blue"]))



    test_sigmoid = np.array([[0, 1], [2, 3]])
    print("Sigmoid for [[0,1], [2,3]]:")
    print("", A2.sigmoid(test_sigmoid))
    print()

    print("Cost for [0,0,0]:")
    print("", A2.logistic_cost(Xne, y, [0, 0, 0]))
    print()

    b = A2.logistic_gradient_descent(Xne, y, 0.5, 1)
    print("B0 = [0,0,0], α = 0.5. After one iteration:")
    print("  B1", b)
    print("  Cost:", A2.logistic_cost(Xne, y, b))
    print()

    beta = A2.logistic_gradient_descent(Xne, y, 0.5, 10000)
    print("Increased iterations:")
    print("  Stable beta:", beta)
    print("  Stable cost:", A2.logistic_cost(Xne, y, beta))
    print()

    grid_size = 1000
    test_grid = grid(Xn[:, 0], Xn[:, 1], grid_size)
    test_grid_e = A2.extend_matrix(test_grid)
    p_grid = A2.sigmoid(np.dot(test_grid_e, beta))
    
    x_min, x_max = Xn[:, 0].min() - 0.1, Xn[:, 0].max() + 0.1
    y_min, y_max = Xn[:, 1].min() - 0.1, Xn[:, 1].max() + 0.1
    plt.figure("Linear decision boundary")
    plt.imshow((p_grid>0.5).reshape(grid_size, grid_size), origin="lower", extent=(x_min, x_max, y_min, y_max), cmap=ListedColormap(["#ffaaaa", "#aaaaff"]))
    plt.scatter(Xn[:, 0], Xn[:, 1], c=y, marker=".", cmap=ListedColormap(["red", "blue"]))


    S = np.array([45, 85])
    Sn = A2.normalize(S, X)
    Sne = A2.extend_matrix(np.array([Sn]))[0]
    prob = A2.sigmoid(np.dot(Sne, beta))
    print("Prob for [45, 85]:", prob)

    p = A2.sigmoid(np.dot(Xne, beta))
    pp = np.round(p)
    print("Training errors:", np.sum(y!=pp))


    plt.show()

if __name__ == "__main__":
    main()