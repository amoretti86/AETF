from __future__ import print_function
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tfa_utils import coordinate_tensor
#from data_generator import DataGenerator


def hot_spot_init(y, k, debug=False):

    # collapse first dimensions (number of images)
    y_fold = np.mean(y, axis=0)

    # get the coordinate tensor
    c = coordinate_tensor(list(y_fold.shape))

    # fold the image
    y_fold = np.abs(y_fold - np.mean(y_fold))

    # # smooth
    # kernel = np.ones([5, 5])
    # kernel /= np.sum(kernel)
    # y_fold = signal.convolve2d(y_fold, kernel, mode='same')

    # get a flat version
    y_flat = y_fold.reshape([-1])

    # initialize center and width lists
    centers = np.zeros([k, len(y_fold.shape)])
    widths = np.zeros(k)

    # loop over the k sources
    for i in range(k):

        print('\rFitting source %d' % (i + 1), end='')

        # find hottest spot as the newest center
        i_hot = np.argmax(y_flat)
        centers[i] = c[i_hot]

        # compute all the squared distances
        dist = np.linalg.norm(centers[i] - np.array(c), axis=1)**2

        # initialize width
        w = 1

        # optimize with Newton's Method
        cost = []
        w_old = w
        for t in range(1000):

            # compute basis image
            y_hat = np.exp(- dist / w)

            # compute cost
            cost.append(0.5 * np.sum((y_hat - y_flat)**2))

            # compute y_hat first derivative w.r.t w
            d1_y_hat = y_hat * (dist * w**(-2))

            # compute cost first derivative w.r.t w
            d1_cost = np.sum((y_hat - y_flat) * d1_y_hat)

            # compute y_hat second derivative w.r.t w
            d2_y_hat = d1_y_hat * (dist * w**(-2)) + y_hat * (-2 * dist * w**(-3))

            # compute cost second derivative w.r.t w
            d2_cost = np.sum(d1_y_hat**2 + y_hat * d2_y_hat - y_flat * d2_y_hat)

            # gradient descent
            w = w - d1_cost / (d2_cost + 1)

            # ensure positive
            if w < 1e-3:
                w = 1e-3

            # check for convergence
            if np.abs(w_old - w) < 1e-5:
                break
            w_old = w

        # save the width as log precision
        widths[i] = -np.log(w)

        # debugging
        if debug:

            # plot the cost
            plt.subplot(3, k, k + 1 + i)
            plt.plot(cost)
            plt.yticks([])

            # label cost if first iteration
            if i == 0:
                plt.ylabel('Cost')

            # plot starting image w/ center and width
            ax = plt.subplot(3, k, 2 * k + 1 + i)
            plt.imshow(y_flat.reshape(y_fold.shape), cmap='jet')
            plt.xticks([])
            plt.yticks([])

            # plot the center/width
            center = np.flip(centers[i], axis=0)
            radius = np.sqrt(np.exp(-widths[i]))
            ax.add_patch(Circle(center, radius, facecolor='None', edgecolor='k'))

            # label image
            if i == 0:
                plt.ylabel('Starting Image')

        # compute basis image
        y_hat = np.exp(- dist * np.exp(widths[i]))

        # subtract basis image from original for next source
        y_flat -= y_hat

    print('\nSources Fit!')

    return centers, widths


if __name__ == '__main__':

    # set number of sources
    k = 5

    # generate the data
    data_gen = DataGenerator(n=5, dim=[30, 30], k=k)
    y = data_gen.generate()

    # loop over the scans
    for i in range(y.shape[0]):

        # generate the subplot for original image
        ax = plt.subplot(3, y.shape[0], i + 1)
        plt.imshow(y[i], cmap='jet')
        plt.xticks([])
        plt.yticks([])

        for j in range(data_gen.k):

            # plot the center/width
            center = (data_gen.centers[j][1], data_gen.centers[j][0])
            radius = np.sqrt(np.exp(-data_gen.widths[j]))
            ax.add_patch(Circle(center, radius, facecolor='None', edgecolor='r'))

        # label image if first iteration
        if i == 0:
            plt.ylabel('Orig. Data')

        # plot the image number
        plt.title('n = %d' % (i + 1))

    # initialize centers and widths
    centers, widths = hot_spot_init(y, k, debug=True)
    plt.show()
