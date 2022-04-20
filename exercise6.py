import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import A2

def forward_selection(X, y, lr):
    """"""
    models = []
    mses = []
    idxs = []
    while len(models) < X.shape[1]:
        mse = []
        k_list = [i for i in range(X.shape[1]) if i not in idxs]
        for k in k_list:
            Xe = X[:, idxs + [k]] if len(idxs) > 0 else X[:, k].reshape(-1, 1)
            lr.fit(Xe, y)
            y_pred = lr.predict(Xe)
            mse.append(mean_squared_error(y, y_pred))
        idxs.append(k_list[np.array(mse).argmin()])
        mses.append(min(mse))
        models.append(np.array(idxs))
    return models, np.array(mses)

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/GPUbenchmark.csv", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]

    lr = LinearRegression()

    uc = np.array([2432, 1607, 1683, 8, 8, 256])
    mdls, means = forward_selection(X, y, lr)
    print(mdls)
    print(means)
    print("Index of model with lowest MSE:", means.argmin())
    print("Model with lowest MSE:", mdls[means.argmin()])
    for m in mdls[1:]:
        lr.fit(X[:, m], y)
        y_pred = lr.predict(uc[m].reshape(1, -1))
        print(y_pred)

if __name__ == "__main__":
    main()