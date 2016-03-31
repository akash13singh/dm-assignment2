import matplotlib.pyplot as plt
import numpy as np

def run():
    x_red=[3,1,2,4,3]
    y_red=[4,4,2,4,1]
    x_red_s_vectors = [2,4]
    y_red_s_vectors = [2,4]
    plt.scatter(x_red, y_red, s=140, c='red', marker=".")
    plt.scatter(x_red_s_vectors, y_red_s_vectors, s = 170, facecolors = 'none', edgecolors='red' )
    x_blue=[4,2,4]
    y_blue=[1,1,3]
    plt.scatter(x_blue, y_blue, s=140, c='blue', marker=".")
    x_blue_s_vectors = [2,4]
    y_blue_s_vectors = [1,3]
    plt.scatter(x_blue_s_vectors, y_blue_s_vectors, s=170, facecolors = 'none', edgecolors='blue')
    plt.xticks([0,1,2,3,4,5])
    plt.yticks([0,1,2,3,4,5])
    sv = np.arange(0,5,.2)
    max_margin_plane,= plt.plot(sv,sv-.5,'g--',label="maximal margin hyperlane")
    margin, = plt.plot(sv,sv,'y--',label="margin")
    plt.plot(sv,sv-1,'y--')
    plt.legend(handles=[max_margin_plane,margin],loc="3")
    plt.show()

if __name__ == "__main__":
    run()