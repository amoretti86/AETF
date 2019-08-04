#from cloud_run import cloud_run

# cloud run?
#if cloud_run():
#    import matplotlib as mpl
#    mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import copy

def log_precision_to_variance(log_pre):
    # var = log(1 / k)
    return np.exp(-log_pre)


def variance_to_log_precision(var):
    # k = log (1 / var)
    return -np.log(var)


def rbf_basis_image(centers, widths, dim):

    # determine number of sources
    k = len(widths)

    # compute coordinate tensor
    coordinates = coordinate_tensor(dim)

    # construct basis image
    f = np.zeros([k, np.prod(dim)])
    for i in range(k):

        # compute distance to source center
        dist = centers[i] - coordinates

        # apply RBF
        f[i] = np.exp(-(np.linalg.norm(dist, axis=1))**2 / log_precision_to_variance(widths[i]))

    # reshape to [k, dim1, ..., dimN]
    f_dim = [k]
    f_dim.extend(dim)
    f = f.reshape(f_dim)

    return f


def plot(y, fig=None, super_title=None, titles=None, plot_dir=None):

    # generate figure if its not supplied
    if fig is None:
        fig = plt.figure()

    # otherwise clear the figure for redrawing
    else:
        fig.clf()

    # get image min and max
    y_min = np.min(y)
    y_max = np.max(y)

    # 2 dimensions
    if len(y.shape[1:]) == 2:

        # loop over the scans
        for i in range(y.shape[0]):

            # generate the subplot
            sp = fig.add_subplot(1, y.shape[0], i+1)
            sp.imshow(y[i], vmin=y_min, vmax=y_max, cmap='jet')
            sp.set_xticks([])
            sp.set_yticks([])
            if titles is not None:
                sp.set_title(titles[i])
            else:
                sp.set_title('n = %d' % (i+1))

    # 3 dimensions
    elif len(y.shape[1:]) == 3:

        # loop over the scans
        for i in range(y.shape[0]):

            # generate the x split subplot
            sp = fig.add_subplot(3, y.shape[0], i + 1)
            sp.imshow(y[i, np.round(y.shape[1] / 2).astype(int), :, :], vmin=y_min, vmax=y_max, cmap='jet')
            sp.set_xticks([])
            sp.set_yticks([])
            if titles is not None:
                sp.set_title('X: ' + titles[i])
            else:
                sp.set_title('X: n = %d' % (i + 1))

            # generate the x split subplot
            sp = fig.add_subplot(3, y.shape[0], i + 1 * y.shape[0] + 1)
            sp.imshow(y[i, :, np.round(y.shape[2] / 2).astype(int), :], vmin=y_min, vmax=y_max, cmap='jet')
            sp.set_xticks([])
            sp.set_yticks([])
            if titles is not None:
                sp.set_title('Y: ' + titles[i])
            else:
                sp.set_title('Y: n = %d' % (i + 1))

            # generate the x split subplot
            sp = fig.add_subplot(3, y.shape[0], i + 2 * y.shape[0] + 1)
            sp.imshow(y[i, :, :, np.round(y.shape[3] / 2).astype(int)], vmin=y_min, vmax=y_max, cmap='jet')
            sp.set_xticks([])
            sp.set_yticks([])
            if titles is not None:
                sp.set_title('Z: ' + titles[i])
            else:
                sp.set_title('Z: n = %d' % (i + 1))

    # super title provided
    if super_title is not None:
        plt.suptitle(super_title)

    # interactive plotting
    if plot_dir is None:
        plt.pause(0.05)

    # saving data
    else:
        fig.savefig(plot_dir + "/" + super_title)
        plt.clf()

    return fig


def coordinate_tensor(dim):
    """
    :param dim: list of lengths for each dimension
    :return: a list of every possible coordinate within the dimension system
    """

    # recurrent function
    def recurrent_dim_filler(dim_current, dim_higher, coordinate_higher, coordinates):

        # processing the lowest dimensions
        if len(dim_current) == 1:

            # loop over the space in this dimension
            for i in range(dim_current[0]):

                # concatenate the higher dimensions' coordinates with the lowest dimension iterator
                new_element = coordinate_higher + [i]

                # add the element to the master coordinate list
                coordinates.append(new_element)

        # still in higher dimensions
        else:

            # pop an element off of the current dimension list onto the higher dimension list
            dim_higher.extend([dim_current.pop(0)])

            # add a placeholder value to the higher dimension coordinate list
            coordinate_higher.extend([-1])

            # loop over the most recently added higher dimension
            for i in range(dim_higher[-1]):

                # update placeholder value
                coordinate_higher[-1] = i

                # recursively call this function
                coordinates = recurrent_dim_filler(dim_current, dim_higher, coordinate_higher, coordinates)

            # reverse the pop
            dim_current.extend([dim_higher.pop(-1)])

            # eliminate place holder value
            coordinate_higher.pop(-1)

        return coordinates

    # initialize lists
    dim_current = copy.deepcopy(list(dim))#list(dim).copy()
    dim_higher = []
    coordinate_higher = []
    coordinates = []

    # run coordinate generator
    coordinates = recurrent_dim_filler(dim_current, dim_higher, coordinate_higher, coordinates)

    return coordinates
