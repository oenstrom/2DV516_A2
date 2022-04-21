import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector

def main():
    """Main function to run when running the script."""
    gender = {"female": 0, "male": 1}
    region = {"northeast": 0, "southeast": 1, "southwest": 2, "northwest": 3}
    feature_names = np.array(["age", "sex", "bmi", "children", "smoker", "region", "shoesize"])
    data = np.array(np.genfromtxt("A2_datasets_2022/insurance.csv", delimiter=";", dtype=None, skip_header=1, encoding="utf-8", case_sensitive="lower",
        converters = { 1: lambda s: gender[s], 4: lambda s: 0 if s == "no" else 1, 5: lambda s: region[s] }).tolist())

    X = data[:, :-1]
    y = data[:, -1]

    plt.figure("Plots of features vs charges", figsize=(12, 7))
    for feat in range(X.shape[1]):
        plt.subplot(3, 3, feat + (1 if feat != 6 else 2))
        plt.gca().set_xlabel(feature_names[feat])
        plt.gca().set_ylabel("Charges")
        plt.scatter(X[:, feat], y, marker=".")
        print(X[:, feat])
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    lir = LinearRegression()
    sfs_forward = SequentialFeatureSelector(lir, n_features_to_select=3, direction="forward").fit(X, y)
    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )

    print()

    rir = Ridge(10000)
    sfs_forward = SequentialFeatureSelector(rir, n_features_to_select=3, direction="forward").fit(X, y)
    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )

    print()

    lar = Lasso(10000)
    sfs_forward = SequentialFeatureSelector(lar, n_features_to_select=3, direction="forward").fit(X, y)
    print(
        "Features selected by forward sequential selection: "
        f"{feature_names[sfs_forward.get_support()]}"
    )

    # X_fs = X[:, sfs_forward.get_support()]
    # lr.fit(X_fs, y)
    # lr.predict()
    # plt.show()

    # for fi in range(X.shape[1]):
    #     Xe = X[:, fi]
    #     lr.fit(Xe.reshape(-1, 1), y)
    #     y_pred = lr.predict(Xe.reshape(-1, 1))
    #     print(f"Feature {fi + 1}: {mean_squared_error(y, y_pred)}")

    # plt.scatter(Xe, y, marker=".")
    # plt.show()

    # linreg = LinearRegression()
    # linreg.fit(X, y)

    # ridreg = Ridge()
    # ridreg.fit(X, y)
    
    # lasreg = Lasso()
    # lasreg.fit(X, y)
    
    # elareg = ElasticNet()
    # elareg.fit(X, y)

    # y_pred_linreg = linreg.predict(X)
    # print(mean_squared_error(y, y_pred_linreg))

    # y_pred_ridreg = ridreg.predict(X)
    # print(mean_squared_error(y, y_pred_ridreg))

    # y_pred_lasreg = lasreg.predict(X)
    # print(mean_squared_error(y, y_pred_lasreg))

    # y_pred_elareg = elareg.predict(X)
    # print(mean_squared_error(y, y_pred_elareg))




if __name__ == "__main__":
    main()