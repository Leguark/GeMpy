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
from GeMpy.GeMpy_core import DataManagement


class PlotData(DataManagement, object):
    """Object Definition to perform Bayes Analysis"""

    def __init__(self, _data, block=None, **kwargs):
        """

        :param _data:
        :param kwds: potential field, block
        """
        self._data = _data
        if block:
            self.block = block

        if 'potential_field':
            self.plot_potential_field = kwargs['potential_field']

    # TODO planning the whole visualization scheme. Only data, potential field and block. 2D 3D? Improving the iteration
    # with pandas framework





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
        kwargs.get('potential_field', True)

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
            plt.ylabel("z")

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
            if hasattr(self, 'potential_field') and "potential_field" in kwargs:
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

        if direction == "x":

            plt.xlabel("y")
            plt.ylabel("z")

            if type(cell_pos) == str:
                # decipher cell position
                if cell_pos == 'center' or cell_pos == 'centre':
                    pos = self.nx / 2
                elif cell_pos == 'min':
                    pos = 0
                elif cell_pos == 'max':
                    pos = self.nx
            else:
                pos = cell_pos

            # Plotting orientations
            plt.quiver(self.dips_position[:, 1], self.dips_position[:, 2], self.G_y, self.G_z, pivot="tail")

            # Plotting interfaces
            if self.layers.ndim == 2:
                layer = self.layers
                plt.plot(layer[:, 1], layer[:, 2], "o")

                if "linear_interpolation" in kwargs:
                    plt.plot(layer[:, 1], layer[:, 2])
            else:
                for layer in self.layers:
                    plt.plot(layer[:, 1], layer[:, 2], "o")

                    if "linear_interpolation" in kwargs:
                        plt.plot(layer[:, 1], layer[:, 2])

            # Plotting potential field if is calculated
            if hasattr(self, 'potential_field') and "potential_field" in kwargs:
                grid_slice = self.potential_field[pos, :, :]
                grid_slice = grid_slice.transpose()
                plt.contour(grid_slice, contour_lines, extent=(self.ymin, self.ymax, self.zmin, self.zmax), **kwargs)
                if colorbar:
                    plt.colorbar()
            # General plot settings
            plt.xlim(self.ymin, self.ymax)
            plt.ylim(self.zmin, self.zmax)
            #  plt.margins(x = 0.1, y = 0.1)
            plt.title("Model Section. Direction: %s. Cell position: %s" % (direction, cell_pos))

class PlotResults(PlotData, object):
    """
    """

    def plot_block_section(self, cell_number=13, **kwargs):

        plot_block = self.block.get_value().reshape(self._data.nx, self._data.ny, self._data.nz)
        plt.imshow(plot_block[:, cell_number, :].T, origin="bottom", aspect="equal",
                   extent=(self.xmin, self.xmax, self.zmin, self.zmax), interpolation="none", **kwargs)


    def plot_potential_field(self, pos, **kwargs):
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
        if hasattr(self, 'potential_field') and "potential_field" in kwargs:
            grid_slice = self.potential_field[:, pos, :]
            grid_slice = grid_slice.transpose()
            plt.contour(grid_slice, extent=(self.xmin, self.xmax, self.zmin, self.zmax), **kwargs)
            if 'colorbar' in kwargs:
                plt.colorbar()
        # General plot settings
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.zmin, self.zmax)
        #  plt.margins(x = 0.1, y = 0.1)
        plt.title("Model Section. Direction: %s. Cell position: %s") #% (direction, cell_pos))

    def export_vtk(self):
        """
        export vtk
        :return:
        """