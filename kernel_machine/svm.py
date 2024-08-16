import sys

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split


def calculate_gaussian_kernel(x_0, x_1, param=0.45):
    return np.exp(-param * np.linalg.norm(x_0 - x_1) ** 2)
    # return np.dot(x_0, x_1)


def create_kernel_matrix(Xs):
    k = np.zeros((Xs.shape[0], Xs.shape[0]))
    for i in range(Xs.shape[0]):
        for j in range(Xs.shape[0]):
            k[i][j] = calculate_gaussian_kernel(Xs[i], Xs[j])

    return k


def calculate_quadratic_programming(x_train, y_train):
    # https://cvxopt.org/userguide/coneprog.html?highlight=qp#quadratic-programming
    print(x_train.shape)
    print(y_train.shape)
    # Calculate parameters(P, q, G, h, A, b)
    k = create_kernel_matrix(x_train)
    P = cvxopt.matrix(np.outer(y_train, y_train) * k)
    print("P: ", P)
    q = cvxopt.matrix(np.ones(x_train.shape[0]) * (-1))
    print("q: ", q)
    G = cvxopt.matrix(np.diag(np.ones(x_train.shape[0]) * (-1)))
    print("G: ", G)
    h = cvxopt.matrix(np.zeros(x_train.shape[0]))
    print("h: ", h)
    A = cvxopt.matrix(y_train.astype("float"), (1, x_train.shape[0]))
    print("A: ", A)
    b = cvxopt.matrix(0.0)
    print("b: ", b)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    a = np.ravel(solution["x"])
    print("a: ", a)

    return a


def calculate_bias(x, y, a):
    b = 0
    for i in range(y.shape[0]):
        w = 0
        for j in range(y.shape[0]):
            t_m = y[j]
            w += a[j] * t_m * calculate_gaussian_kernel(x[i], x[j])

        b += y[i] - w

    b /= y.shape[0]
    print("b: ", b)

    return b


def predict(x_train, y_train, x_test, a, b):
    pred = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        for j in range(x_train.shape[0]):
            pred[i] += (
                a[j] * y_train[j] * calculate_gaussian_kernel(x_train[j], x_test[i])
            )

        pred[i] += b

    return np.sign(pred)


def main():
    # we create 40 separable points
    x, y = make_blobs(n_samples=50, centers=2, random_state=6)
    y[y == 0] = -1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    a = calculate_quadratic_programming(x_train, y_train)
    # Calculate bias
    b = calculate_bias(x_train, y_train, a)
    # Predict test data by using parameters
    pred = predict(x_train, y_train, x_test, a, b)
    print("pred: ", pred)
    print("y_test: ", y_test)

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(x, y)
    print(y)
    input()
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=pred, s=30, cmap=plt.cm.Paired)

    plt.show()
    input()
    print(x)
    print(y)

    # plot the decision function
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        x,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
