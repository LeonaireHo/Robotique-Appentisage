#!/usr/local/bin/python

import numpy as np
import time
from rbfn import RBFN
from lwr import LWR
from line import Line
from sample_generator import SampleGenerator
import matplotlib.pyplot as plt

class Main:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.batch_size = 50

    def reset_batch(self):
        self.x_data = []
        self.y_data = []

    def make_nonlinear_batch_data(self):
        """ 
        Generate a batch of non linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def make_linear_batch_data(self):
        """ 
        Generate a batch of linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def approx_linear_batch(self):
        model = Line(self.batch_size)

        self.make_linear_batch_data()

        # start = time.process_time()
        # model.train(self.x_data, self.y_data)
        # print("LLS time:", time.process_time() - start)
        # model.plot(self.x_data, self.y_data)
        #
        # start = time.process_time()
        # model.train_from_stats(self.x_data, self.y_data)
        # print("LLS from scipy stats:", time.process_time() - start)
        # model.plot(self.x_data, self.y_data)

        # start = time.process_time()
        # model.train_regularized(self.x_data, self.y_data, coef=0.01)
        # print("regularized LLS :", time.process_time() - start)
        # model.plot(self.x_data, self.y_data)

        idx = []
        res = []
        for i in range(500):
            idx.append(i)
            model.train_regularized(self.x_data, self.y_data, coef=0.01 * i)
            res.append(model.residuals(self.x_data, self.y_data))
        plt.plot(idx, res)
        plt.show()

    def approx_rbfn_batch_test(self,nb_features):

        model = RBFN(nb_features)
        self.make_nonlinear_batch_data()

        # start = time.process_time()
        # model.train_ls(self.x_data, self.y_data)
        # print("RBFN LS time:", time.process_time() - start)
        # model.plot(self.x_data, self.y_data)

        start = time.process_time()
        model.train_ls2(self.x_data, self.y_data)
        print("RBFN LS2 time:", time.process_time() - start)
        # model.plot(self.x_data, self.y_data)
        return nb_features,model.residuals(self.x_data,self.y_data)

    def approx_rbfn_batch(self):

        model = RBFN(nb_features=10)
        self.make_nonlinear_batch_data()

        start = time.process_time()
        model.train_ls(self.x_data, self.y_data)
        print("RBFN LS time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

        start = time.process_time()
        model.train_ls2(self.x_data, self.y_data)
        print("RBFN LS2 time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

    def approx_rbfn_iterative_Test_Iter(self):
        model = RBFN(50)
        start = time.process_time()
        # Generate a batch of data and store it
        self.reset_batch()
        g = SampleGenerator()
        idx = []
        res = []
        for i in range(20,1000):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

            # Comment the ones you don't want to use
            # model.train_gd(x, y, alpha=0.5)
            model.train_rls(x, y)
            # model.train_rls_sherman_morrison(x, y)
            residuals = model.residuals(self.x_data, self.y_data)
            idx.append(i)
            print(residuals)
            res.append(residuals/i)
        plt.plot(idx,res)
        plt.show()

    def approx_rbfn_iterative_TestNB(self):
        idx = []
        res = np.zeros((10))
        for x in range(200):
            for i in range(10):
                max_iter = 100
                model = RBFN(15)
                start = time.process_time()
                # Generate a batch of data and store it
                self.reset_batch()
                g = SampleGenerator()
                for j in range(max_iter):
                    # Draw a random sample on the interval [0,1]
                    x = np.random.random()
                    y = g.generate_non_linear_samples(x)
                    self.x_data.append(x)
                    self.y_data.append(y)

                    # Comment the ones you don't want to use
                    model.train_gd(x, y, alpha = i * 0.02 + 1)
                    #model.train_rls(x, y)
                    # model.train_rls_sherman_morrison(x, y)
                res[i] += model.residuals(self.x_data, self.y_data)
        for i in range(10):
            idx.append(i * 0.02 + 1)
            res[i] /= 200
        plt.plot(idx, res,color = "red")
        plt.show()
        print("RBFN Incr time:", time.process_time() - start)
        # model.plot(self.x_data, self.y_data)
    def approx_rbfn_iterative(self):
        residuals = 0
        t = 0
        for i in range(100):
            max_iter = 500
            model = RBFN(nb_features=15)
            start = time.process_time()
            # Generate a batch of data and store it
            self.reset_batch()
            g = SampleGenerator()
            for i in range(max_iter):
                # Draw a random sample on the interval [0,1]
                x = np.random.random()
                y = g.generate_non_linear_samples(x)
                self.x_data.append(x)
                self.y_data.append(y)

                # Comment the ones you don't want to use
                # model.train_gd(x, y, alpha=0.5)
                # model.train_rls(x, y)
                model.train_rls_sherman_morrison(x, y)
                # if(i == 500):
                #     model.plot(self.x_data, self.y_data)
            print("RBFN Incr time:", time.process_time() - start)
            t += time.process_time() - start
            residuals += model.residuals(self.x_data, self.y_data)
        print(residuals/100,t/100)
        model.plot(self.x_data, self.y_data)
    def approx_lwr_batch(self):
        model = LWR(nb_features=10)
        self.make_nonlinear_batch_data()

        start = time.process_time()
        model.train_lwls(self.x_data, self.y_data)
        print("LWR time:", time.process_time() - start)
        model.plot(self.x_data, self.y_data)

if __name__ == '__main__':
    m = Main()
    # m.approx_linear_batch()
    m.approx_rbfn_batch()
    # m.approx_rbfn_iterative()
    # m.approx_lwr_batch()
    print("Fini")

    #
    # iters = []
    # nb = []
    # temps = []
    # rdl =[]
    # idx = []
    # nb_test_iters = 1000
    #
    # for i in range(nb_test_iters):
    #     res = ( m.approx_rbfn_iterative_Test_Iter(i*2,20))
    #     iters.append(res[0])
    #     temps.append(res[2])
    #     rdl.append(res[3])
    # rdl = np.zeros((20))
    # for j in range(100):
    #     for i in range(1,20):
    #         _,res = m.approx_rbfn_batch_test(i)
    #     # temps.append(res[2])
    #         rdl[i] += res
    # for i in range(20):
    #     rdl[i] /= 20
    #     nb.append(i)
    # # plt.plot(idx, iters, lw=2, color='red')
    # # plt.plot(nb, temps[:100], lw=2, color='blue')
    # plt.plot(nb[1:], rdl[1:], lw=2, color='green')
    # plt.show()
