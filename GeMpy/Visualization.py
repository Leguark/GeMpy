"""
Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 23/09/2016

@author: Miguel de la Varga
"""

import numpy as np
import matplotlib.pyplot as plt

# TODO: inherit pygeomod classes
import sys, os
sys.path.append("/home/bl3/PycharmProjects/pygeomod/")
#from home/bl3/PycharmProjects/pygeomod/pygeomod import geogrid

class GeoPlot():
    """Object Definition to perform Bayes Analysis"""

    def __init__(self, **kwds):
        """nothing"""

    def set_basename(self, name):
        """Set basename for grid exports, etc.

        **Arguments**:
            - *name* = string: basename
        """
        self.basename = name

    def set_extent(self,x_min, x_max, y_min, y_max, z_min, z_max):
        """Set basename for grid exports, etc.

            **Arguments**:
                - *3-D coordinates* = float: dimensions of the domain
            """
        self.xmin = x_min
        self.xmax = x_max
        self.ymin = y_min
        self.ymax = y_max
        self.zmin = z_min
        self.zmax = z_max

    def set_resolutions(self, nx, ny, nz):
        """

        :param nx: Resolution in x direction
        :param ny: Resolution in y direction
        :param nz: Resolution in z direction
        :return: The self values
        """

        self.nx ,self.ny, self.nz = nx, ny, nz

    def set_layers(self, layers):

        """ Generate a 3-D dimensional array where:
         - axis 0 contains xyz coordinates of every points.
         - axis 1 every point per layer
         - axis 2 every layer
        :param layers: 2D numpy array of every layer
        :return: 3D numpy array encapsulating every layer
        """

        self.layers = np.asarray(layers)

    def calculate_gradient(self):
        """ Calculate the gradient vector of module 1 given dip and azimuth

        :return: Components xyz of the unity vector.
        """
        self.G_x = np.sin(np.deg2rad(self.dips_angles)) * np.sin(np.deg2rad(self.azimuths)) * self.polarity
        self.G_z = np.cos(np.deg2rad(self.dips_angles)) * self.polarity
        self.G_y = np.sin(np.deg2rad(self.dips_angles)) * np.cos(np.deg2rad(self.azimuths)) * self.polarity

    def plot_potential_field_2D(self, direction="x", cell_pos="center", **kwargs):

        """Plot a section through the model in a given coordinate direction

                **Arguments**:
                    - *direction* = 'x', 'y', 'z' : coordinate direction for section position

                **Optional Kwargs**:
                    - *cmap* = mpl.colormap : define colormap for plot (default: jet)
                    - *alpha* = define the transparencie for the plot (default: 1)
                    - *colorbar* = bool: attach colorbar (default: True)
                    - *geomod_coord* = bool:  Plotting geomodeller coordinates instead voxels (default: False)
                    - *contour* = bool : Plotting contour of the layers contact
                    - *plot layer* = array: Layer Number you want to plot in the contour plot
                    - *rescale* = bool: rescale color bar to range of visible slice (default: False)
                    - *ve* = float : vertical exageration (for plots in x,y-direction)
                    - *figsize* = (x,y) : figsize settings for plot
                    - *ax* = matplotlib.axis : add plot to this axis (default: new axis)
                            if axis is defined, the axis is returned and the plot not shown
                            Note: if ax is passed, colorbar is False per default!
                    - *savefig* = bool : save figure to file (default: show)
                    - *fig_filename* = string : filename to save figure
        """

        # TODO: include the kwargs into the matplotlib functions

        colorbar = kwargs.get('colorbar', True)
        cmap = kwargs.get('cmap', 'jet')
        alpha = kwargs.get('alpha', 1)
        rescale = kwargs.get('rescale', False)
        ve = kwargs.get('ve', 1.)
        figsize = kwargs.get('figsize', (8, 4))
        geomod_coord = kwargs.get('geomod_coord', False)
        contour = kwargs.get('contour', False)
        contour_lines = kwargs.get('contour_lines', 10)
        linewidth = kwargs.get("linewidth", 1)
        levels = kwargs.get("plot_layer", None)
        kwargs.get("linear_interpolation", False)

        if not "ax" in kwargs:
            colorbar = kwargs.get('colorbar', True)
            # create new axis for plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            colorbar = False
            ax = kwargs['ax']

        # TODO: Add the other two directions x and z
        if direction == "y":

            plt.xlabel("x")
            plt.ylabel("y")

            if type(cell_pos) == str:
                # decipher cell position
                if cell_pos == 'center' or cell_pos == 'centre':
                    pos = self.ny / 2
                elif cell_pos == 'min':
                    pos = 0
                elif cell_pos == 'max':
                    pos = self.ny
            else:
                pos = cell_pos



            # Plotting orientations
            plt.quiver(self.dips_position[:, 0], self.dips_position[:, 2], self.G_x, self.G_z, pivot="tail")

            # Plotting interfaces
            if self.layers.ndim == 2:
                layer = self.layers
                plt.plot(layer[:, 0], layer[:, 2], "o")

                if "linear_interpolation" in kwargs:
                    plt.plot(layer[:, 0], layer[:, 2])
            else:
                for layer in self.layers:
                    plt.plot(layer[:, 0], layer[:, 2], "o")

                    if "linear_interpolation" in kwargs:
                        plt.plot(layer[:, 0], layer[:, 2])

            # Plotting potential field if is calculated
            if hasattr(self, 'potential_field'):
                grid_slice = self.potential_field[:, pos, :]
                grid_slice = grid_slice.transpose()
                plt.contour(grid_slice, contour_lines, extent=(self.xmin, self.xmax, self.zmin, self.zmax), **kwargs)
                if colorbar:
                    plt.colorbar()
            # General plot settings
            plt.xlim(self.xmin, self.xmax)
            plt.ylim(self.zmin, self.zmax)
          #  plt.margins(x = 0.1, y = 0.1)
            plt.title("Model Section. Direction: %s. Cell position: %s" % (direction,cell_pos))