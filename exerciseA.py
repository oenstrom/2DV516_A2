import numpy as np
import matplotlib.pyplot as plt

def extend_matrix(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def normal_eq(Xe, y):
    return np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

def normalize(x_to_norm, X):
    return (x_to_norm - np.mean(X, axis=0, dtype=np.float64)) / np.std(X, axis=0, dtype=np.float64)

def cost(Xe, y, beta):
    j = np.dot(Xe, beta)-y
    return (j.T.dot(j))/Xe.shape[0]

# def gradient_descent(X, y, a = 0.0002, n = 20000000):
def gradient_descent(X, y, a = 0.01, n = 1000, plot = False):
    """Perform gradient descent on the given X"""
    w = np.zeros(X.shape[1])
    costs = []
    for i in range(n):
        j = (X.T).dot(X.dot(w) - y)
        w = w - (a * j) / X.shape[0]
        costs.append(cost(X, y, w))
    if plot:
        plt.figure()
        plt.plot(range(n), costs)
    return w


def main():
    data = np.loadtxt("A2_datasets_2022/girls_height.csv")

    X = data[:, 1:3]
    y = data[:, 0]

    plt.figure("Girl, mom")
    plt.scatter(X[:, 0], y, marker=".", c="None", edgecolors="black")
    plt.xlabel("mom height")
    plt.ylabel("girl height")

    plt.figure("Girl, dad")
    plt.xlabel("dad height")
    plt.ylabel("girl height")
    plt.scatter(X[:, 1], y, marker=".", c="None", edgecolors="black")

    Xe = extend_matrix(X)
    beta = normal_eq(Xe, y)

    print(beta)
    print(cost(Xe, y, beta))
    ug1 = [1, 65, 70]

    print(np.dot(ug1, beta))

    Xn = normalize(X, X)
    plt.figure("Girl, mom, Feature normalization")
    plt.scatter(Xn[:, 0], y, marker=".", c=[[0,0,0,0]], edgecolors="black")

    plt.figure("Girl, dad, Feature normalization")
    plt.scatter(Xn[:, 1], y, marker=".", c=[[0,0,0,0]], edgecolors="black")

    Xne = extend_matrix(Xn)
    beta_n = normal_eq(Xne, y)
    print(beta_n)
    print(cost(Xne, y, beta_n))
    ug1_n = normalize(np.array([65, 70]), X)
    print(np.dot(np.append([1], ug1_n), beta_n))
    # exit()
    print()
    print()
    print()

    # Gradient descent on non-normalized X
    # res = gradient_descent(Xe, y, a = 0.0002, n = 20000000)
    
    # Gradient descent on normalized X
    beta_gradient = gradient_descent(Xne, y, a = 0.05, n = 200, plot = True)
    print("Normalization, gradient descent:")
    print("  Height:", np.dot(np.append([1], ug1_n), beta_gradient))
    print("  Cost: ", cost(Xne, y, beta_gradient))




    plt.show()
    # print(cost(Xe, y, beta))


if __name__ == "__main__":
    main()