import numpy as np

def mapFeature(X1, X2, D):
    """From lecture 5: Logistic Regression, modified slightly.

    Creates polynomial model of degree D of the given features X1 and X2."""
    Xe = np.c_[np.ones((len(X1), 1)), X1, X2]
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
    b = np.zeros(X.shape[1])
    costs = []
    for _ in range(n):
        b = b - (a*X.T.dot((sigmoid(X.dot(b)) - y)))/X.shape[0]
        costs.append(logistic_cost(X, y, b))
    return (b, costs) if with_costs else b