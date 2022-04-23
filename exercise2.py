import numpy as np
import matplotlib.pyplot as plt
import A2

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/housing_price_index.csv", delimiter=",")
    X = data[:, 0]
    y = data[:, 1]

    # Task 1
    plt.figure("Original housing price index data")
    plt.plot(X + 1975, y, ".-")
    plt.xlabel("Year")
    plt.ylabel("Index")

    # Task 2
    plt.figure("Polynomial models degree 1-4", figsize=(12, 7))
    for i in range(1, 4+1):
        Xe = A2.extend_matrix(X, i)
        B = A2.normal_eq(Xe, y)
        plt.subplot(2, 2, i)
        plt.gca().set_title(f"cost = {np.round(A2.cost(Xe, y, B), 2)}")
        plt.plot(X + 1975, y, ".-", label="Original data")
        plt.plot(X + 1975, np.dot(Xe, B), label=f"Degree = {i}")
        plt.xlabel("Year")
        plt.ylabel("Index")
        plt.legend()
    plt.tight_layout()

    # Task 3
    Xe     = A2.extend_matrix(X, 4)
    B      = A2.normal_eq(Xe, y)
    X_test = A2.extend_matrix(np.array([47]), 4)[0]

    predicted_idx_2022 = np.dot(X_test, B)
    idx_2015 = y[2015 - 1975]
    inc = predicted_idx_2022 / idx_2015
    price_2022 = 2.3 * inc
    print("Predicted price for 2022:", np.round(price_2022, 6), "million SEK")




if __name__ == "__main__":
    main()
    plt.show()