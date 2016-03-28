import matplotlib.pyplot as plt
import numpy as np

def run():
    x_red=[3,2,4,1]
    y_red=[4,2,4,4]
    x_red_s_vectors = [2,4]
    y_red_s_vectors = [2,4]
    plt.scatter(x_red, y_red, s=80, c='red', marker=".")
    plt.scatter(x_red_s_vectors, y_red_s_vectors, s=80, c='red', marker=">")
    x_blue=[2,4,4]
    y_blue=[1,3,1]
    plt.scatter(x_blue, y_blue, s=80, c='blue', marker=".")
    x_blue_s_vectors = [2,4]
    y_blue_s_vectors = [1,3]
    plt.scatter(x_blue_s_vectors, y_blue_s_vectors, s=80, c='blue', marker=">")
    plt.xticks([0,1,2,3,4,5])
    plt.yticks([0,1,2,3,4,5])
    sv = np.arange(0,5,.2)
    plt.plot(sv,sv-.5,'g--')
    plt.show()

if __name__ == "__main__":
    run()