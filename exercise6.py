import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import A2

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/GPUbenchmark.csv", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]


    models = A2.forward_selection(X, y, as_indexes=False)
    lr = LinearRegression()
    cross_mses = []
    for fs in range(1, len(models) + 1):
        m = models[:fs] - 1
        y_pred = cross_val_predict(lr, X[:, m], y, cv=3)
        mse = mean_squared_error(y, y_pred)
        cross_mses.append(mse)
        print(f"{len(m)} feature(s) model MSE: {mse}")
    cross_mses = np.array(cross_mses)
    print(f"Best model: {models[:cross_mses.argmin() + 1]}")
    print(f"Most important feature: {models[0]}")


if __name__ == "__main__":
    main()