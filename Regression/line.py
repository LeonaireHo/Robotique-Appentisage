
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class Line:
    def __init__(self, batch_size):
        self.nb_dims = 1
        self.theta = np.zeros(shape=(batch_size, self.nb_dims+1))

    def f(self, x):
        """
        Get the FA output for a given input variable(s)

        :param x: A single or vector of dependent variables with size [Ns] for which to calculate the features

        :returns: the function approximator output
        """
        if np.size(x) == 1:
            xl = np.vstack(([x], [1]))
        else:
            xl = np.vstack((x, np.ones((1, np.size(x)))))
        return np.dot(self.theta, xl)

    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train(self, x_data, y_data):
        # Finds the Least Square optimal weights
        x_data = np.array([x_data]).transpose()
        y_data = np.array(y_data)
        x = np.hstack((x_data, np.ones((x_data.shape[0], 1))))

    #TODO: Fill this
        self.theta = np.linalg.inv(x.T @ x) @ x.T @ y_data
        # ----------------------#
        # # Training Algorithm ##
        # ----------------------#

    def train_regularized(self, x_data, y_data, coef):
        # Finds the regularized Least Square optimal weights
        x_data = np.array([x_data]).transpose()
        y_data = np.array(y_data)
        x = np.hstack((x_data, np.ones((x_data.shape[0], 1))))

        #TODO: Fill this
        self.theta =np.linalg.inv((coef * np.identity(x.T.shape[0])) + x.T @ x) @ x.T @ y_data
    def residuals(self, x_data, y_data):
        # x_data = np.array([x_data]).T
        # y_data = np.array(y_data)
        # d = np.array([y_data]).T - x_data * self.theta[:-1] - self.theta[-1]
        d =np.array([self.f(x_data[i]) - y_data[i] for i in range(len(x_data))])
        return np.sum(d**2)
    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train_from_stats(self, x_data, y_data):
        # Finds the Least Square optimal weights: python provided version
        slope, intercept, r_value, _, _ = stats.linregress(x_data, y_data)

        #TODO: Fill this

    # -----------------#
    # # Plot function ##
    # -----------------#

    def plot(self, x_data, y_data):
        xs = np.linspace(0.0, 1.0, 1000)
        z = self.f(xs)
        print(z)
        plt.plot(x_data, y_data, 'o', markersize=3, color='lightgreen')
        plt.plot(xs, z, lw=2, color='red')
        plt.show()
