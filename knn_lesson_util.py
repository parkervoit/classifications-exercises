import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()

COLORS = {"pistachio": "darkgreen", "blueberry": "blue", "chocolate": "saddlebrown"}


def get_ice_cream_data(pre_scale=False):
    #: Generate the fake data and split it
    np.random.seed(123)
    df = pd.DataFrame(
        {
            "flavor": np.random.choice(["pistachio", "blueberry", "chocolate"], 100),
            "pints": np.random.normal(10, 1, 100),
            "n_sprinkles": np.random.normal(1000, 300, 100).round(),
        }
    )

    df.pints = np.where(
        df.flavor == "pistachio", np.random.normal(13, 1, 100), df.pints
    )
    df.pints = np.where(df.flavor == "blueberry", np.random.normal(7, 1, 100), df.pints)
    df.n_sprinkles = np.where(
        df.flavor == "pistachio", np.random.normal(1080, 300, 100), df.n_sprinkles
    )
    df.n_sprinkles = np.where(
        df.flavor == "blueberry", np.random.normal(920, 300, 100), df.n_sprinkles
    )

    if pre_scale:
        df.n_sprinkles /= 100
    
    train, test = train_test_split(df, train_size=0.9)
    
    return train, test


#: Visualize KNN with just pints
def plot_pints(train, test=None, plot_test=False):
    for flavor, pints in train.groupby("flavor").pints:
        plt.scatter(pints, [0] * pints.size, label=flavor, c=COLORS[flavor], alpha=0.5)
    plt.xlabel("Pints Consumed")
    plt.ylim(-0.1, 0.2)
    plt.yticks([])
    plt.ylabel("")
    plt.gca().spines["left"].set_visible(False)
    if plot_test:
        plt.scatter(
            test.pints, [0.05] * test.pints.size, c="black", marker="x", label="???"
        )
    plt.legend()


#: Visualize KNN with pints and sprinkles, demo why scaling matters
def plot_pints_and_sprinkles(train, test, plot_test=False, same_xy_scale=False):
    for flavor, subset in train.groupby("flavor"):
        plt.scatter(
            subset.pints, subset.n_sprinkles, label=flavor, c=COLORS[flavor], alpha=0.5
        )
    plt.xlabel("Pints Consumed")
    plt.ylabel("# Sprinkles")
    if plot_test:
        plt.scatter(
            test.pints, test.n_sprinkles, c="black", marker="x", label="???", s=128
        )
    plt.legend()
    if same_xy_scale:
        plt.xlim(-300, 300)


def get_complex_data():
    np.random.seed(123)
    n = 1000
    df = pd.DataFrame(
        {
            "flavor": np.random.choice(["pistachio", "blueberry", "chocolate"], n),
            "pints": np.random.normal(10, 1, n),
            "n_sprinkles": np.random.normal(1000, 100, n).round(),
        }
    )
    df.pints = np.where(
        df.flavor == "pistachio", np.random.normal(12, 0.5, n), df.pints
    )
    df.pints = np.where(df.flavor == "blueberry", np.random.normal(8, 0.8, n), df.pints)
    df.n_sprinkles = np.where(
        df.flavor == "pistachio", np.random.normal(1100, 50, n), df.n_sprinkles
    )
    df.n_sprinkles = np.where(
        df.flavor == "blueberry", np.random.normal(900, 50, n), df.n_sprinkles
    )
    colors = {"pistachio": "darkgreen", "blueberry": "red", "chocolate": "brown"}
    train, test = train_test_split(df)

    return train, test


def eval_knn_model(k, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    return {
        "k": k,
        "train_accuracy": knn.score(X_train, y_train),
        "test_accuracy": knn.score(X_test, y_test),
    }


def plot_k_vs_accuracy(X_train, y_train, X_test, y_test, max_k=25):
    models = [
        eval_knn_model(k, X_train, y_train, X_test, y_test) for k in range(1, max_k + 1)
    ]
    df = pd.DataFrame(models)
    return df.set_index("k").plot()


def plot_misses(df):
    for flavor, subset in df.groupby("flavor"):
        plt.scatter(
            subset.pints, subset.n_sprinkles, c=COLORS[flavor], marker="o", alpha=0.5, label=flavor
        )
    for flavor, subset in df[~df.correct].groupby("flavor"):
        plt.scatter(
            subset.pints, subset.n_sprinkles, c=COLORS[flavor], marker="x", s=256,
            label=f'Actually {flavor}, incorrect prediction'
        )
    plt.gca().set(ylabel='n_sprinkles', xlabel='pints')
    plt.legend()
