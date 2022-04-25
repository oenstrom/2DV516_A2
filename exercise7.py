import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector

def plot_r2(estimators, X_train, y_train, X_test, y_test, max_c=10001, step=5):
    """Plot a graph of λ vs R2 for the different estimators."""
    cs = list(range(1, max_c + 1, step))
    for r_name, r in estimators.items():
        scores = []
        for c in cs:
            if r_name != "Linear":
                r.alpha = c
            r.fit(X_train, y_train)
            y_pred = r.predict(X_test)
            scores.append(r2_score(y_test, y_pred))
        plt.plot(cs, scores, label=r_name)
    plt.legend()
    plt.xlabel("λ (alpha)")
    plt.ylabel("r2 score")

def main():
    """Main function to run when running the script."""
    gender = {"female": 0, "male": 1}
    region = {"northeast": 0, "southeast": 1, "southwest": 2, "northwest": 3}
    feature_names = np.array(["age", "sex", "bmi", "children", "smoker", "region", "shoesize"])
    data = np.array(np.genfromtxt("A2_datasets_2022/insurance.csv", delimiter=";", dtype=None, skip_header=1, encoding="utf-8", case_sensitive="lower",
        converters = { 1: lambda s: gender[s], 4: lambda s: 0 if s == "no" else 1, 5: lambda s: region[s] }).tolist())

    X = data[:, :-1]
    y = data[:, -1]
    train_end = round(len(X)*0.80)
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]


    plt.figure("Plots of features vs charges", figsize=(12, 7))
    for feat in range(X.shape[1]):
        plt.subplot(3, 3, feat + (1 if feat != 6 else 2))
        plt.gca().set_xlabel(feature_names[feat])
        plt.gca().set_ylabel("Charges")
        plt.scatter(X[:, feat], y, marker=".")
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    lir, rir, lar, enr = LinearRegression(), Ridge(), Lasso(), ElasticNet()
    ests = {"Linear": lir, "Ridge": rir, "Lasso": lar, "ElasticNet": enr}

    plt.figure("R2 scores for different estimators")
    plot_r2(ests, X_train, y_train, X_test, y_test)


    sfs_forward = SequentialFeatureSelector(lir, n_features_to_select=3, direction="forward").fit(X_train, y_train)
    print(f"Normal Linear:    {feature_names[sfs_forward.get_support()]}")
    features_to_use = sfs_forward.get_support()

    rir.alpha, lar.alpha, enr.alpha = 3000, 3000, 3000
    sfs_forward = SequentialFeatureSelector(rir, n_features_to_select=3, direction="forward").fit(X_train, y_train)
    print(f"Ridge   λ = {rir.alpha}: {feature_names[sfs_forward.get_support()]}")

    sfs_forward = SequentialFeatureSelector(lar, n_features_to_select=3, direction="forward").fit(X_train, y_train)
    print(f"Lasso   λ = {lar.alpha}: {feature_names[sfs_forward.get_support()]}")

    sfs_forward = SequentialFeatureSelector(enr, n_features_to_select=3, direction="forward").fit(X_train, y_train)
    print(f"Elastic λ = {enr.alpha}: {feature_names[sfs_forward.get_support()]}")


    X_train_fs = X_train[:, features_to_use]
    X_test_fs = X_test[:, features_to_use]

    plt.figure("R2 scores for different estimators, with forward selection")
    plot_r2(ests, X_train_fs, y_train, X_test_fs, y_test)


    X_train_fs = np.c_[X_train_fs, X_train_fs[:, 0]**4, X_train_fs[:, 1]**3, X_train_fs[:, 2]**2]
    X_test_fs = np.c_[X_test_fs, X_test_fs[:, 0]**4, X_test_fs[:, 1]**3, X_test_fs[:, 2]**2]
    rir.alpha, lar.alpha, enr.alpha = 3, 3, 3
    for e_name, est in ests.items():
        print(e_name)
        est.fit(X_train_fs, y_train)
        y_pred = est.predict(X_test_fs)
        print("  R2: ", r2_score(y_test, y_pred))



if __name__ == "__main__":
    main()
    # plt.show()