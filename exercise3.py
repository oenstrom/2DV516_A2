import numpy as np
import matplotlib.pyplot as plt
import A2

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/breast_cancer.csv", delimiter=",")
    # np.random.seed(19950516)
    np.random.shuffle(data)
    data[:, -1] = np.where(data[:, -1] == 2, 0, 1)
    training_set = data[:546]
    test_set = data[546:]

    X_train   = training_set[:, :-1]
    y_train   = training_set[:, -1]
    Xn_train  = A2.normalize(X_train, X_train)
    Xne_train = A2.extend_matrix(Xn_train)

    X_test   = test_set[:, :-1]
    y_test   = test_set[:, -1]
    Xn_test  = A2.normalize(X_test, X_train)
    Xne_test = A2.extend_matrix(Xn_test)

    #[-1.17778759  1.38587477 -0.28414295  1.37469494  1.09701026  0.20807157
    #  1.34080335  0.85543409  0.49236138  0.95227544]

    alpha, N = 5.0, 2000
    beta, costs = A2.logistic_gradient_descent(Xne_train, y_train, alpha, N, with_costs=True)
    print("Hyperparameters:")
    print(f"  α = {alpha}\n  N = {N}")
    print()

    plt.figure("Cost over iterations", figsize=(12, 7))
    plt.suptitle(f"α = {alpha}, N = {N}  (50 first rows skipped in plot)")
    plt.xlabel("Iterations (N)")
    plt.ylabel("Cost (J)")
    plt.plot(range(N)[50:], costs[50:])


    ############################################################################
    ## Exercise 3: Task 4                                                     ##
    ############################################################################
    p = A2.sigmoid(np.dot(Xne_train, beta))
    pp = np.round(p)
    print("Training errors:", np.sum(y_train!=pp))
    print("Training accuracy:", np.round(100*np.sum(y_train==pp)/np.size(pp), 5), "%")
    print()


    ############################################################################
    ## Exercise 3: Task 5                                                     ##
    ############################################################################
    p_test = A2.sigmoid(np.dot(Xne_test, beta))
    pp_test = np.round(p_test)
    print("Test errors:", np.sum(y_test!=pp_test))
    print("Test accuracy:", np.round(100*np.sum(y_test==pp_test)/np.size(pp_test), 5), "%")


    # Yes they are qualitatively the same. There are only small differences in both the training accuracy and test accuracy.
    # Yes they depend on how many observations I put aside for testing. At 20% test, the test accuracy were really close
    # to the training accuracy and many times above.
    # At 50% test, the test accuracy went down slightly and were often below the training accuracy.
    # At 90% test, there are just not enough training data. The test accuracy is really unreliable, with accuracies sometimes under 88%.
    # IT DOESN'T AFFECT IT SO MUCH THOUGH! WRITE SOMETHING BETTER ABOUT IT!


if __name__ == "__main__":
    main()
    # plt.show()