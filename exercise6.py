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
            # print(Xe)
            lr.fit(Xe, y)
            y_pred = lr.predict(Xe)
            mse.append(mean_squared_error(y, y_pred))
        idxs.append(k_list[np.array(mse).argmin()])
        mses.append(min(mse))
        models.append(np.array(idxs))
    return models, mses

def main():
    """Main function to run when running the script."""
    data = np.loadtxt("A2_datasets_2022/GPUbenchmark.csv", delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]

    # print(X.shape)

    lr = LinearRegression()
    lr.fit(X, y)

    uc = np.array([2432, 1607, 1683, 8, 8, 256])
    res = lr.predict(uc.reshape(1, -1))
    print(res)
    print(mean_squared_error([114], res))
    hej = lr.predict(X)
    print("MSE:", mean_squared_error(y, hej))
    print()
    print("-----------------------------------------")


    # lr.fit(X[:, 0].reshape(-1, 1), y)
    # y_pred = lr.predict(X[:, 0].reshape(-1, 1))
    # print(y_pred)
    # print("MSE 1:", mean_squared_error(y, y_pred))
    # print()

    # lr.fit(X[:, 1].reshape(-1, 1), y)
    # y_pred = lr.predict(X[:, 1].reshape(-1, 1))
    # print(y_pred)
    # print("MSE F2:", mean_squared_error(y, y_pred))
    # print()

    # lr.fit(X[:, 2].reshape(-1, 1), y)
    # y_pred = lr.predict(X[:, 2].reshape(-1, 1))
    # print(y_pred)
    # print("MSE F3:", mean_squared_error(y, y_pred))
    # print()

    # lr.fit(X[:, 3].reshape(-1, 1), y)
    # y_pred = lr.predict(X[:, 3].reshape(-1, 1))
    # print(y_pred)
    # print("MSE F4:", mean_squared_error(y, y_pred))
    # print()

    # lr.fit(X[:, 4].reshape(-1, 1), y)
    # y_pred = lr.predict(X[:, 4].reshape(-1, 1))
    # print(y_pred)
    # print("MSE F5:", mean_squared_error(y, y_pred))
    # print()

    # lr.fit(X[:, 5].reshape(-1, 1), y)
    # y_pred = lr.predict(X[:, 5].reshape(-1, 1))
    # print(y_pred)
    # print("MSE F6:", mean_squared_error(y, y_pred))
    # print()
    # print("---------------------------------------------")
    # print()

    # mdls = []
    # mse_1 = []
    # for k in range(X.shape[1]):
    #     lr.fit(X[:, k].reshape(-1, 1), y)
    #     y_pred = lr.predict(X[:, k].reshape(-1, 1))
    #     mse_1.append(mean_squared_error(y, y_pred))
    # min_idx = np.array(mse_1).argmin()
    # mdls.append([min_idx])

    # for k2 in [i for i in range(X.shape[1]) if i not in [min_idx]]:
    #     print(k2)


    models = []
    mses = []
    idxs = []
    while len(models) < X.shape[1]:
        mse = []
        k_list = [i for i in range(X.shape[1]) if i not in idxs]
        for k in k_list:
            Xe = X[:, idxs + [k]] if len(idxs) > 0 else X[:, k].reshape(-1, 1)
            # print(Xe)
            lr.fit(Xe, y)
            y_pred = lr.predict(Xe)
            mse.append(mean_squared_error(y, y_pred))
        idxs.append(k_list[np.array(mse).argmin()])
        mses.append(min(mse))
        models.append(np.array(idxs))
        # print(idxs)
        # print(models)
    # print(np.array(mse).argmin())
    # print(mse)
    print(models)
    print(mses)

    uc = np.array([2432, 1607, 1683, 8, 8, 256])
    # res = lr.predict(uc.reshape(1, -1))
    # print(res)
    # print(mean_squared_error([114], res))
    for m in models[1:]:
        lr.fit(X[:, m], y)
        y_res = lr.predict(uc[m].reshape(1, -1))
        print(y_res)
        # y_res = lr.predict(X[:, m])
        # print(mean_squared_error(y, y_res))
    
    print("USING FUNCTION")
    mdls, means = forward_selection(X, y, lr)
    print(mdls)
    print(mses)
    for m in mdls[1:]:
        lr.fit(X[:, m], y)
        y_res = lr.predict(uc[m].reshape(1, -1))
        print(y_res)

if __name__ == "__main__":
    main()