import numpy as np
import matplotlib.pyplot as plt
import A2

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/GPUBenchmark.csv", delimiter=",")

    X = data[:, :-1]
    y = data[:, -1]
    Xn = A2.normalize(X, X) # Exercise 1: Task 1
    Xne = A2.extend_matrix(Xn)

    ############################################################################
    ## Exercise 1: Task 2                                                     ##
    ############################################################################
    plt.figure("Features and result", figsize=(12, 7))
    for i in range(Xn.shape[1]):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(Xn[:, i], y, marker=".", c="None", edgecolors="black")
        plt.xlabel(f"Feature {i+1}")
        plt.ylabel("Result")
        plt.xlim([-3, 3])
    plt.tight_layout()


    ############################################################################
    ## Exercise 1: Task 3                                                     ##
    ############################################################################
    beta = A2.normal_eq(Xne, y)
    uc = np.array([2432, 1607, 1683, 8, 8, 256])
    uc_n = A2.normalize(uc, X)
    print(f"3. Predicted benchmark for {uc}:")
    print("     Result:", np.dot(np.append([1], uc_n), beta))
    print()

    ############################################################################
    ## Exercise 1: Task 4                                                     ##
    ############################################################################
    print("4. Cost using Beta computed by the normal equation: ")
    print("     Cost:", A2.cost(Xne, y, beta))
    print()

    ############################################################################
    ## Exercise 1: Task 5                                                     ##
    ############################################################################
    alpha, iterations = 0.4, 3000
    beta_grad = A2.gradient_descent(Xne, y, a = alpha, n = iterations)
    print("5. Gradient descent:")
    print("  a)")
    print(f"     α = {alpha}, N = {iterations}")
    print("     Cost: ", A2.cost(Xne, y, beta_grad))
    print("  b)")
    print("     Result:", np.dot(np.append([1], uc_n), beta_grad))


    plt.show()

if __name__ == "__main__":
    main()