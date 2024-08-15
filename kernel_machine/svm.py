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


def calculate_quadratic_programming(X_train, y_train):
    # https://cvxopt.org/userguide/coneprog.html?highlight=qp#quadratic-programming
    print(X_train.shape)
    print(y_train.shape)
    # Calculate parameters(P, q, G, h, A, b)
    k = create_kernel_matrix(X_train)
    P = cvxopt.matrix(np.outer(y_train, y_train) * k)
    print("P: ", P)
    q = cvxopt.matrix(np.ones(X_train.shape[0]) * (-1))
    print("q: ", q)
    G = cvxopt.matrix(np.diag(np.ones(X_train.shape[0]) * (-1)))
    print("G: ", G)
    h = cvxopt.matrix(np.zeros(X_train.shape[0]))
    print("h: ", h)
    A = cvxopt.matrix(y_train.astype("float"), (1, X_train.shape[0]))
    print("A: ", A)
    b = cvxopt.matrix(0.0)
    print("b: ", b)
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    a = np.ravel(solution["x"])
    print("a: ", a)

    return a


def main():
    # we create 40 separable points
    X, y = make_blobs(n_samples=50, centers=2, random_state=6)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    a = calculate_quadratic_programming(X_train, y_train)
    input()

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    print(X)
    print(y)

    # plot the decision function
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
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
