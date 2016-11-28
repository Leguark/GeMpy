"""
Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 23/09/2016

@author: Miguel de la Varga
"""


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# TODO: inherit pygeomod classes
#import sys, os



class PlotData(object):
    """Object Definition to perform Bayes Analysis"""

    def __init__(self, _data, block=None, **kwargs):
        """

        :param _data:
        :param kwds: potential field, block
        """
        self._data = _data
        if block:
            self._block = block

        if 'potential_field' in kwargs:
            self._potential_field_p = kwargs['potential_field']

    # TODO planning the whole visualization scheme. Only data, potential field and block. 2D 3D? Improving the iteration
    # with pandas framework
        self._set_style()

    def _set_style(self):
        plt.style.use(['seaborn-white', 'seaborn-paper'])
        matplotlib.rc("font", family="Times New Roman")

    def plot_data(self, direction="y", serie="all", **kwargs):
        """
        Plot the projection of all data
        :param direction:
        :return:
        """
        x, y, Gx, Gy = self._slice(direction)[4:]

        if serie == "all":
            series_to_plot_i = self._data.Interfaces[self._data.Interfaces["series"].
                                                     isin(self._data.series.columns.values)]
            series_to_plot_f = self._data.Foliations[self._data.Foliations["series"].
                                                     isin(self._data.series.columns.values)]

        else:
            series_to_plot_i = self._data.Interfaces[self._data.Interfaces["series"] == serie]
            series_to_plot_f = self._data.Foliations[self._data.Foliations["series"] == serie]
        sns.lmplot(x, y,
                   data=series_to_plot_i,
                   fit_reg=False,
                   hue="formation",
                   scatter_kws={"marker": "D",
                                "s": 100},
                   legend=True,
                   legend_out=True,
                   **kwargs)

        # Plotting orientations
        plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                   series_to_plot_f[Gx], series_to_plot_f[Gy],
                   pivot="tail")

        plt.xlabel(x)
        plt.ylabel(y)

    def _slice(self, direction, cell_number=25):
        _a, _b, _c = slice(0, self._data.nx), slice(0, self._data.ny), slice(0, self._data.nz)
        if direction == "x":
            _a = cell_number
            x = "Y"
            y = "Z"
            Gx = "G_y"
            Gy = "G_z"
            extent_val = self._data.ymin, self._data.ymax, self._data.zmin, self._data.zmax
        elif direction == "y":
            _b = cell_number
            x = "X"
            y = "Z"
            Gx = "G_x"
            Gy = "G_z"
            extent_val = self._data.xmin, self._data.xmax, self._data.zmin, self._data.zmax
        elif direction == "z":
            _c = cell_number
            x = "X"
            y = "Y"
            Gx = "G_x"
            Gy = "G_y"
            extent_val = self._data.xmin, self._data.xmax, self._data.ymin, self._data.ymax
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def plot_block_section(self, cell_number=13, direction="y", **kwargs):

        plot_block = self._block.get_value().reshape(self._data.nx, self._data.ny, self._data.nz)
        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        plt.imshow(plot_block[_a, _b, _c].T, origin="bottom", aspect="equal", cmap="viridis",
                   extent=extent_val,
                   interpolation="none", **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)

    def plot_potential_field(self, cell_number, potential_field=None, n_pf=0,
                             direction="y", plot_data=True, serie="all", **kwargs):

        if not potential_field:
            potential_field = self._potential_field_p[n_pf]

        if plot_data:
            self.plot_data(direction, self._data.series.columns.values[n_pf])

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
        plt.contour(potential_field[_a, _b, _c].T,
                    extent=extent_val,
                    **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.title(self._data.series.columns[n_pf])
        plt.xlabel(x)
        plt.ylabel(y)

    def export_vtk(self):
        """
        export vtk
        :return:
        """