import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

def mapFeature(X1, X2, D, ones = True):
    """From lecture 5: Logistic Regression, modified slightly.

    Creates polynomial model of degree D of the given features X1 and X2."""
    Xe = np.c_[np.ones((len(X1), 1)), X1, X2] if ones else np.c_[X1, X2]
    for i in range(2,D+1):
        for j in range(0,i+1):
            Xnew = X1**(i-j)*X2**j
            Xe = np.c_[Xe, Xnew]
    return Xe

def extend_matrix(X, D = 1):
    """Extend the given matrix X to the polynomial of the given degree D."""
    Xe = np.c_[np.ones((X.shape[0], 1)), X]
    for i in range(2, D+1):
        Xe = np.c_[Xe, X**i]
    return Xe

def normal_eq(Xe, y):
    """Calculate betas using the normal equation."""
    return np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

def normalize(x_to_norm, X):
    """Normalize the matrix x_to_norm using the matrix X."""
    return (x_to_norm - np.mean(X, axis=0, dtype=np.float64)) / np.std(X, axis=0, dtype=np.float64)

def cost(Xe, y, beta):
    """Vectorized mean squared distance."""
    j = np.dot(Xe, beta)-y
    return (j.T.dot(j))/Xe.shape[0]

def gradient_descent(X, y, a = 0.01, n = 1000):
    """Perform gradient descent on the given X"""
    w = np.zeros(X.shape[1])
    for _ in range(n):
        j = (X.T).dot(X.dot(w) - y)
        w = w - (a * j) / X.shape[0]
    return w

def sigmoid(X):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-X))

def logistic_cost(X, y, beta):
    """Logistic cost"""
    ep = 1e-5
    return -(y.T.dot(np.log(sigmoid(X.dot(beta)) + ep)) + ((1-y).T).dot(np.log(1 - sigmoid(X.dot(beta)) + ep)))/X.shape[0]

def logistic_gradient_descent(X, y, a = 0.01, n = 1000, with_costs = False):
    """Perform logistic gradient descent on the given X"""
    b = np.zeros(X.shape[1])
    costs = []
    for _ in range(n):
        b = b - (a*X.T.dot((sigmoid(X.dot(b)) - y)))/X.shape[0]
        costs.append(logistic_cost(X, y, b))
    return (b, costs) if with_costs else b

def predict_probability(X, beta):
    """Use sigmoid to predict the probability of X using the provided betas."""
    return sigmoid(np.dot(X, beta))

def forward_selection(X, y, cross_val=False, as_indexes=False):
    """Forward selection on X using MSE to select the best features."""
    lr = LinearRegression()
    idxs = []
    while len(idxs) < X.shape[1]:
        mse = []
        k_list = [i for i in range(X.shape[1]) if i not in idxs]
        for k in k_list:
            Xe = X[:, idxs + [k]] if len(idxs) > 0 else X[:, k].reshape(-1, 1)
            lr.fit(Xe, y)
            # y_pred = lr.predict(Xe)
            y_pred = lr.predict(Xe) if not cross_val else cross_val_predict(lr, Xe, y, cv=3)
            mse.append(mean_squared_error(y, y_pred))
        idxs.append(k_list[np.array(mse).argmin()])
    return np.array(idxs) if as_indexes else (np.array(idxs) + 1)