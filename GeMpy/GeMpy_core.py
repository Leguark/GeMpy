"""
Module with classes and methods to perform implicit regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 10/10 /2016

@author: Miguel de la Varga

"""

import theano
import theano.tensor as T
import numpy as np
import sys, os
import pandas as pn

from Visualization import PlotData


class GeMpy(object):
    def __init__(self, project_name='Default'):
        # nothing
        self.project_name = project_name
        # TODO it has to be something better than None for this
        self.Data = None
        self.Grid = None
        self.Interpolator = None
        self.Plot = None

    def import_data(self, *args, **kwargs):
        self.Data = DataManagement(*args, **kwargs)
        self.Plot = PlotData(self.Data)

    def create_grid(self, grid_type="regular_3D"):
        self.Grid = Grid(self.Data, grid_type=grid_type)

    def set_interpolator(self):
        self.Interpolator = Interpolator(self.Data, self.Grid)
        self.Plot = PlotData(self.Data, block=self.Interpolator.block,
                             potential_field=self.Interpolator.potential_fields)

    def update_data(self, *args, **kwargs):
        self.Data = DataManagement(*args, **kwargs)
        self.Plot = PlotData(self.Data, block=self.Interpolator.block,
                             potential_field=self.Interpolator.potential_field)


class DataManagement(object):
    """
    Class to import the raw data of the model

    """

    # TODO: Data management using pandas, find an easy way to add values
    # TODO: Probably at some point I will have to make an static and dynamic data classes
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max,
                 nx=50, ny=50, nz=50,
                 path_i=os.getcwd(), path_f=os.getcwd()
                 ):
        """
        Initial parameters of the project
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :param z_min:
        :param z_max:
        :param nx:
        :param ny:
        :param nz:
        :param path_i:
        :param path_f:
        """

        self.xmin = x_min
        self.xmax = x_max
        self.ymin = y_min
        self.ymax = y_max
        self.zmin = z_min
        self.zmax = z_max

        self.nx = nx
        self.ny = ny
        self.nz = nz

        # TODO choose the default source of data. So far only
        self.Foliations = self.load_data_csv(data_type="foliations", path=path_f)
        self.Interfaces = self.load_data_csv(data_type="Interfaces", path=path_i)

        assert set(['X', 'Y', 'Z', 'formation']).issubset(self.Interfaces.columns), \
            "One or more columns do not match with the expected values " + str(self.Interfaces.columns)
        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(self.Foliations.columns), \
            "One or more columns do not match with the expected values " + str(self.Foliations.columns)

        self.formations = self.set_formations()
        self.series = self.set_series()
        self.calculate_gradient()

    @staticmethod
    def load_data_csv(data_type, path=os.getcwd(), **kwargs):

        # TODO: in case that the columns have a different name specify in pandas which columns are interfaces /
        #  coordinates, dips and so on.
        # TODO: use pandas to read any format file not only csv
        """
        Method to load either interface or foliations data csv files. Normally this is in which GeoModeller exports it
        :param data_type: string, 'interfaces' or 'foliations'
        :param path: path to the files
        :return: Pandas framework with the imported data
        """
        if data_type == "foliations":
            return pn.read_csv(path, **kwargs)
        else:
            return pn.read_csv(path, **kwargs)

        # TODO if we load different data the Interpolator parameters must be also updated:  Research how and implement

    def set_formations(self):
        try:
            getattr(self, "formations")
        except AttributeError:
            try:
                # Foliations may or may not be in all formations so we need to use Interfaces
                formations = self.Interfaces["formation"].unique()

                # TODO: Trying to make this more elegant?
                for el in self.formations:
                    for check in self.formations:
                        assert (el not in check or el == check), "One of the formations name contains other" \
                                                                 " string. Please rename." + str(el) + " in " + str(
                            check)

                        # TODO: Add the possibility to change the name in pandas directly
                        # (adding just a 1 in the contained string)
            except AttributeError:
                pass
        return formations

    def set_series(self, series_distribution=None, order=None):
        """
        The formations have to be separated by this thing! |
        :param series_distribution:
        :param order of the series by default takes the dictionary keys which until python 3.6 are random
        :return:
        """
        if series_distribution is None:
            # TODO: Possibly we have to debug this function
            _series = {"Default serie": self.formations}

        else:
            assert type(series_distribution) is dict, "series_distribution must be a dictionary, " \
                                                      "see Docstring for more information"
            _series = series_distribution
        if not order:
            order = _series.keys()
        _series = pn.DataFrame(data=_series, columns=order)
        assert np.count_nonzero(np.unique(_series.values)) is len(self.formations), \
            "series_distribution must have the same number of values as number of formations %s." \
            % self.formations

        self.Interfaces["series"] = [(i == _series).sum().argmax() for i in self.Interfaces["formation"]]
        self.Foliations["series"] = [(i == _series).sum().argmax() for i in self.Foliations["formation"]]

        self.series = _series
        return _series

    def calculate_gradient(self):
        """ Calculate the gradient vector of module 1 given dip and azimuth

        :return: Components xyz of the unity vector.
        """

        self.Foliations['G_x'] = np.sin(np.deg2rad(self.Foliations["dip"])) * \
            np.sin(np.deg2rad(self.Foliations["azimuth"])) * self.Foliations["polarity"]
        self.Foliations['G_y'] = np.sin(np.deg2rad(self.Foliations["dip"])) * \
            np.cos(np.deg2rad(self.Foliations["azimuth"])) * self.Foliations["polarity"]
        self.Foliations['G_z'] = np.cos(np.deg2rad(self.Foliations["dip"])) * self.Foliations["polarity"]

    def get_spatial_parameters(self):
        return self.xmax, self.xmin, self.ymax, self.ymin, self.zmax, self.zmin, self.nx, self.ny, self.nz

    # TODO set new interface/set


class Grid(object):
    """
    Class with set of functions to generate grids
    """
    def __init__(self, spatial_parameters, grid_type="regular_3D"):
        """
        Selection of grid type
        :param type: So far regular 3D or 2D grid
        """
        self._grid_par = spatial_parameters

        if grid_type == "regular_3D":
            self.grid = self.create_regular_grid_3d()
        elif grid_type == "regular_2D":
            self.grid = self.create_regular_grid_2d()
        else:
            print("Wrong type")

    def create_regular_grid_2d(self):
        """
        Method to create a 2D regular grid where we interpolate
        :return: 2D regular grid for the resoulution nx, ny
        """
        try:
            g = np.meshgrid(
                np.linspace(self._grid_par.xmin, self._grid_par.xmax, self._grid_par.nx, dtype="float32"),
                np.linspace(self._grid_par.ymin, self._grid_par.ymax, self._grid_par.ny, dtype="float32"),
            )
            return np.vstack(map(np.ravel, g)).T.astype("float32")
        except AttributeError:
            print("Extent or resolution not provided. Use set_extent and/or set_resolutions first")

    def create_regular_grid_3d(self):
        """
        Method to create a 3D regurlar grid where is interpolated
        :return: 3D regurlar grid for the resolution nx,ny
        """

        g = np.meshgrid(
            np.linspace(self._grid_par.xmin, self._grid_par.xmax, self._grid_par.nx, dtype="float32"),
            np.linspace(self._grid_par.ymin, self._grid_par.ymax, self._grid_par.ny, dtype="float32"),
            np.linspace(self._grid_par.zmin, self._grid_par.zmax, self._grid_par.nz, dtype="float32"), indexing="ij"
        )

        return np.vstack(map(np.ravel, g)).T.astype("float32")

# class DataPreparation(object):
#     """
#     Class that takes raw data and converts it into theano parameters
#     """
#     def __init__(self, _data, _grid, range_var=None, c_o=None, nugget_effect=0.01, u_grade=2, rescaling_factor=None):
#         """
#         Basic interpolator parameters. Also here it is possible to change some flags of theano
#         :param range_var: Range of the variogram, it is recommended the distance of the longest diagonal
#         :param c_o: Sill of the variogram
#         """
#         # TODO: update Docstring
#
#         theano.config.optimizer = 'None'
#         theano.config.exception_verbosity = 'high'
#         theano.config.compute_test_value = 'ignore'
#
#         if not range_var:
#             range_var = np.sqrt((_data.xmax-_data.xmin)**2 +
#                                 (_data.ymax-_data.ymin)**2 +
#                                 (_data.zmax-_data.zmin)**2)
#         if not c_o:
#             c_o = range_var**2/14/3
#
#         # Creation of shared variables
#         self.nx_T = theano.shared(_data.nx, "Resolution in x axis")
#         self.ny_T = theano.shared(_data.ny, "Resolution in y axis")
#         self.nz_T = theano.shared(_data.nz, "Resolution in z axis")
#         self.a_T = theano.shared(range_var, "range", allow_downcast=True)
#         self.c_o_T = theano.shared(c_o, "covariance at 0", allow_downcast=True)
#         self.nugget_effect_grad_T = theano.shared(nugget_effect, "nugget effect of the grade", allow_downcast=True)
#         assert (0 <= u_grade <= 2)
#         if u_grade == 0:
#             self.u_grade_T = theano.shared(u_grade, "grade of the universal drift", allow_downcast=True)
#         else:
#             self.u_grade_T = theano.shared(3**u_grade, allow_downcast=True)
#         # TODO: To be sure what is the mathematical meaning of this
#
#         if not rescaling_factor:
#             max_coord = pn.concat([_data.Foliations, _data.Interfaces]).max()[:3]
#             min_coord = pn.concat([_data.Foliations, _data.Interfaces]).min()[:3]
#             rescaling_factor = np.max(max_coord - min_coord)
#
#         self.rescaling_factor_T = theano.shared(rescaling_factor, "rescaling factor", allow_downcast=True)
#         self.n_formations_T = theano.shared(np.zeros(2), "Vector. Number of formations in the serie")
#
#         _universal_matrix = np.vstack((_grid.grid.T,
#                                        (_grid.grid ** 2).T,
#                                        _grid.grid[:, 0] * _grid.grid[:, 1],
#                                        _grid.grid[:, 0] * _grid.grid[:, 2],
#                                        _grid.grid[:, 1] * _grid.grid[:, 2]))
#         self.universal_matrix_T = theano.shared(_universal_matrix + 1e-10, "universal matrix")
#
#         self.block = theano.shared(np.zeros_like(_grid.grid[:, 0]), "Final block")
#         self.grid_val_T = theano.shared(_grid.grid + 1e-10, "Positions of the points to interpolate")


class Interpolator(object):
    """
    Class which contain all needed methods to perform potential field implicit modelling
    """
    def __init__(self, _data, _grid, *args):#range_var=None, c_o=None, nugget_effect=0.01, u_grade=2, rescaling_factor=None):

        theano.config.optimizer = 'fast_compile'
        theano.config.exception_verbosity = 'high'
        theano.config.compute_test_value = 'ignore'

        self._data = _data
        self._grid = _grid

        self._set_constant_parameteres(_data, _grid, *args)
        self.theano_compilation_3D()

        self.potential_fields = [self.compute_potential_field(i)for i in np.arange(len(self._data.series.columns))]

    def _select_serie(self, series_name=0):
        """
        Return the formations of a given serie in string
        :param series_name: name or argument of the serie. Default first of the list
        :return: formations of a given serie in string separeted by |
        """
        if type(series_name) == int or type(series_name) == np.int64:
            _formations_in_serie = "|".join(self._data.series.ix[:, series_name].drop_duplicates())
        elif type(series_name) == str:
            _formations_in_serie = "|".join(self._data.series[series_name].drop_duplicates())
        return _formations_in_serie

    def _set_constant_parameteres(self, _data, _grid, range_var=None, c_o=None,
                                  nugget_effect=0.01, u_grade=2, rescaling_factor=None):
        """
               Basic interpolator parameters. Also here it is possible to change some flags of theano
               :param range_var: Range of the variogram, it is recommended the distance of the longest diagonal
               :param c_o: Sill of the variogram
               """
        # TODO: update Docstring

        if not range_var:
            range_var = np.sqrt((_data.xmax - _data.xmin) ** 2 +
                                (_data.ymax - _data.ymin) ** 2 +
                                (_data.zmax - _data.zmin) ** 2)
        if not c_o:
            c_o = range_var ** 2 / 14 / 3

        # Creation of shared variables
        self.nx_T = theano.shared(_data.nx, "Resolution in x axis")
        self.ny_T = theano.shared(_data.ny, "Resolution in y axis")
        self.nz_T = theano.shared(_data.nz, "Resolution in z axis")
        self.a_T = theano.shared(range_var, "range", allow_downcast=True)
        self.c_o_T = theano.shared(c_o, "covariance at 0", allow_downcast=True)
        self.nugget_effect_grad_T = theano.shared(nugget_effect, "nugget effect of the grade", allow_downcast=True)
        assert (0 <= u_grade <= 2)
        if u_grade == 0:
            self.u_grade_T = theano.shared(u_grade, "grade of the universal drift", allow_downcast=True)
        else:
            self.u_grade_T = theano.shared(3 ** u_grade, allow_downcast=True)
        # TODO: To be sure what is the mathematical meaning of this

        if not rescaling_factor:
            max_coord = pn.concat([_data.Foliations, _data.Interfaces]).max()[:3]
            min_coord = pn.concat([_data.Foliations, _data.Interfaces]).min()[:3]
            rescaling_factor = np.max(max_coord - min_coord)

        self.rescaling_factor_T = theano.shared(rescaling_factor, "rescaling factor", allow_downcast=True)
        self.n_formations_T = theano.shared(np.zeros(2), "Vector. Number of formations in the serie")

        _universal_matrix = np.vstack((_grid.grid.T,
                                       (_grid.grid ** 2).T,
                                       _grid.grid[:, 0] * _grid.grid[:, 1],
                                       _grid.grid[:, 0] * _grid.grid[:, 2],
                                       _grid.grid[:, 1] * _grid.grid[:, 2]))
        self.universal_matrix_T = theano.shared(_universal_matrix + 1e-10, "universal matrix")

        self.block = theano.shared(np.zeros_like(_grid.grid[:, 0]), "Final block")
        self.grid_val_T = theano.shared(_grid.grid + 1e-10, "Positions of the points to interpolate")

    def compute_block_model(self, series_number="all", verbose=0):

        if series_number == "all":
            series_number = np.arange(len(self._data.series.columns))
        for i in series_number:
            formations_in_serie = self._select_serie(i)
            n_formation = np.squeeze(np.where(np.in1d(self._data.formations, self._data.series.ix[:, i])))+1
            if verbose > 0:
                print(n_formation)
            self._aux_computations_block_model(formations_in_serie, np.array(n_formation, ndmin=1),
                                               verbose=verbose)

    def compute_potential_field(self, series_name=0, verbose=0):

        assert series_name is not "all", "Compute potential field only returns one potential field at the time"
        formations_in_serie = self._select_serie(series_name)
        return self._aux_computations_potential_field(formations_in_serie, verbose=verbose)

    def _aux_computations_block_model(self, for_in_ser, n_formation, verbose=0):

        # TODO Probably here I should add some asserts for sanity check
        try:
            yet_simulated = (self.block.get_value() == 0) * 1
            if verbose > 0:
                print(yet_simulated, (yet_simulated == 0).sum())
        except AttributeError:
            yet_simulated = np.ones_like(self._data.grid[:, 0], dtype="int8")
            print("I am in the except")
        # TODO: change [:,:3] that is positional based for XYZ so is more consistent
        dips_position = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)].as_matrix()[:, :3]
        dip_angles = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)]["dip"].as_matrix()
        azimuth = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)]["azimuth"].as_matrix()
        polarity = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)]["polarity"].as_matrix()

        if for_in_ser.count("|") == 0:
            layers = self._data.Interfaces[self._data.Interfaces["formation"].str.contains(for_in_ser)].as_matrix()[:, :3]
            rest_layer_points = layers[1:]
            # TODO self.n_formation probably should not be self
            self.n_formations_T.set_value(np.array(rest_layer_points.shape[0], ndmin=1))
            ref_layer_points = np.tile(layers[0], (np.shape(layers)[0] - 1, 1))
        else:
            # TODO: This is ugly
            layers_list = []
            for formation in for_in_ser.split("|"):
                layers_list.append(self._data.Interfaces[self._data.Interfaces["formation"] == formation].as_matrix()[:, :3])
            layers = np.asarray(layers_list)

            rest_layer_points = layers[0][1:]
            rest_dim = np.array(layers[0][1:].shape[0], ndmin=1)
            for i in layers[1:]:
                rest_layer_points = np.vstack((rest_layer_points, i[1:]))
                rest_dim = np.append(rest_dim, rest_dim[-1] + i[1:].shape[0])
            self.n_formations_T.set_value(rest_dim)
            ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))

        if verbose > 0:
            print("The serie formations are %s" % for_in_ser)
            if verbose > 1:
                print("The formations are: \n"
                      "Layers ", self._data.Interfaces[self._data.Interfaces["formation"].str.contains(for_in_ser)], " \n "
                                                                                                         "Foliations ",
                      self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)])

        # self.grad is none so far. I have it for further research in the calculation of the Jacobian matrix
        self.grad = self._block_export(dips_position, dip_angles, azimuth, polarity,
                                      rest_layer_points, ref_layer_points,
                                      n_formation, yet_simulated)

    def _aux_computations_potential_field(self,  for_in_ser, verbose=0):

        # TODO: change [:,:3] that is positional based for XYZ so is more consistent
        dips_position = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)].as_matrix()[:, :3]
        dip_angles = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)]["dip"].as_matrix()
        azimuth = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)]["azimuth"].as_matrix()
        polarity = self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)]["polarity"].as_matrix()

        if for_in_ser.count("|") == 0:
            layers = self._data.Interfaces[self._data.Interfaces["formation"].str.contains(for_in_ser)].as_matrix()[:, :3]
            rest_layer_points = layers[1:]
            ref_layer_points = np.tile(layers[0], (np.shape(layers)[0] - 1, 1))
        else:
            layers_list = []
            for formation in for_in_ser.split("|"):
                layers_list.append(self._data.Interfaces[self._data.Interfaces["formation"] == formation].as_matrix()[:, :3])
            layers = np.asarray(layers_list)
            rest_layer_points = np.vstack((i[1:] for i in layers))
            ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))

        if verbose > 0:
            print("The serie formations are %s" % for_in_ser)
            if verbose > 1:
                print("The formations are: \n"
                      "Layers ", self._data.Interfaces[self._data.Interfaces["formation"].str.contains(for_in_ser)], " \n "
                                                                                                    "Foliations ",
                      self._data.Foliations[self._data.Foliations["formation"].str.contains(for_in_ser)])

        self.Z_x, G_x, G_y, G_z, self.potential_interfaces, C, DK = self._interpolate(
            dips_position, dip_angles, azimuth, polarity,
            rest_layer_points, ref_layer_points)[:]

        potential_field = self.Z_x.reshape(self._data.nx, self._data.ny, self._data.nz)

        if verbose > 2:
            print("Gradients: ", G_x, G_y, G_z)
            print("Dual Kriging weights: ", DK)
        if verbose > 3:
            print("C_matrix: ", C)

        return potential_field

    def theano_compilation_3D(self):
        """
        Function that generates the symbolic code to perform the interpolation
        :return: Array containing the potential field (maybe it returs all the pieces too I have to evaluate how
        this influences performance)
        """

        # Creation of symbolic variables
        dips_position = T.matrix("Position of the dips")
        dip_angles = T.vector("Angle of every dip")
        azimuth = T.vector("Azimuth")
        polarity = T.vector("Polarity")
        ref_layer_points = T.matrix("Reference points for every layer")
        rest_layer_points = T.matrix("Rest of the points of the layers")

        # Init values
        n_dimensions = 3
        grade_universal = self.u_grade_T

        # Calculating the dimensions of the
        length_of_CG = dips_position.shape[0] * n_dimensions
        length_of_CGI = rest_layer_points.shape[0]
        length_of_U_I = grade_universal
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I

        # Extra parameters
        i_reescale = 1 / (self.rescaling_factor_T ** 2)
        gi_reescale = 1 / self.rescaling_factor_T

        # TODO: Check that the distances does not go nuts when I use too large numbers

        # Here we create the array with the points to simulate:
        #   Grid points except those who have been simulated in a younger serie
        #   Interfaces points to segment the lithologies
        yet_simulated = T.vector("boolean function that avoid to simulate twice a point of a different serie")
        grid_val = T.vertical_stack((
            self.grid_val_T*yet_simulated.reshape((yet_simulated.shape[0], 1))).nonzero_values().reshape((-1, 3)),
                                    rest_layer_points)

        # ==========================================
        # Calculation of Cartesian and Euclidian distances
        # ===========================================
        # Auxiliary tile for dips and transformation to float 64 of variables in order to calculate precise euclidian
        # distances
        _aux_dips_pos = T.tile(dips_position, (n_dimensions, 1)).astype("float64")
        _aux_rest_layer_points = rest_layer_points.astype("float64")
        _aux_ref_layer_points = ref_layer_points.astype("float64")
        _aux_grid_val = grid_val.astype("float64")

        # Calculation of euclidian distances giving back float32
        SED_rest_rest = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_rest_layer_points.T))).astype("float32")

        SED_ref_rest = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_rest_layer_points.T))).astype("float32")

        SED_rest_ref = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_ref_layer_points.T))).astype("float32")

        SED_ref_ref = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_ref_layer_points.T))).astype("float32")

        SED_dips_dips = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_dips_pos ** 2).sum(1).reshape((1, _aux_dips_pos.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_dips_pos.T))).astype("float32")

        SED_dips_rest = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_rest_layer_points.T))).astype("float32")

        SED_dips_ref = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_ref_layer_points.T))).astype("float32")

        # Calculating euclidian distances between the point to simulate and the avalible data
        SED_dips_SimPoint = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_grid_val.T))).astype("float32")

        SED_rest_SimPoint = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_grid_val.T))).astype("float32")

        SED_ref_SimPoint = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_grid_val.T))).astype("float32")

        # Cartesian distances between dips positions
        h_u = T.vertical_stack(
            T.tile(dips_position[:, 0] - dips_position[:, 0].reshape((dips_position[:, 0].shape[0], 1)), n_dimensions),
            T.tile(dips_position[:, 1] - dips_position[:, 1].reshape((dips_position[:, 1].shape[0], 1)), n_dimensions),
            T.tile(dips_position[:, 2] - dips_position[:, 2].reshape((dips_position[:, 2].shape[0], 1)), n_dimensions))

        h_v = h_u.T

        # Cartesian distances between dips and interface points
        # Rest
        hu_rest = T.vertical_stack(
            (dips_position[:, 0] - rest_layer_points[:, 0].reshape((rest_layer_points[:, 0].shape[0], 1))).T,
            (dips_position[:, 1] - rest_layer_points[:, 1].reshape((rest_layer_points[:, 1].shape[0], 1))).T,
            (dips_position[:, 2] - rest_layer_points[:, 2].reshape((rest_layer_points[:, 2].shape[0], 1))).T
        )

        # Reference point
        hu_ref = T.vertical_stack(
            (dips_position[:, 0] - ref_layer_points[:, 0].reshape((ref_layer_points[:, 0].shape[0], 1))).T,
            (dips_position[:, 1] - ref_layer_points[:, 1].reshape((ref_layer_points[:, 1].shape[0], 1))).T,
            (dips_position[:, 2] - ref_layer_points[:, 2].reshape((ref_layer_points[:, 2].shape[0], 1))).T
        )

        # Cartesian distances between reference points and rest
        hx = T.stack(
            (rest_layer_points[:, 0] - ref_layer_points[:, 0]),
            (rest_layer_points[:, 1] - ref_layer_points[:, 1]),
            (rest_layer_points[:, 2] - ref_layer_points[:, 2])
        ).T

        # Cartesian distances between the point to simulate and the dips
        hu_SimPoint = T.vertical_stack(
            (dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
            (dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
            (dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        )

        # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        # every gradient direction covariance
        perpendicularity_matrix = T.zeros_like(SED_dips_dips)

        # Cross-covariances of x
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[0:dips_position.shape[0], 0:dips_position.shape[0]], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[dips_position.shape[0]:dips_position.shape[0] * 2,
            dips_position.shape[0]:dips_position.shape[0] * 2], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[dips_position.shape[0] * 2:dips_position.shape[0] * 3,
            dips_position.shape[0] * 2:dips_position.shape[0] * 3], 1)

        # ==========================
        # Creating covariance Matrix
        # ==========================
        # Covariance matrix for interfaces
        C_I = (self.c_o_T * i_reescale * (
            (SED_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (SED_rest_rest / self.a_T) ** 2 +
             35 / 4 * (SED_rest_rest / self.a_T) ** 3 -
             7 / 2 * (SED_rest_rest / self.a_T) ** 5 +
             3 / 4 * (SED_rest_rest / self.a_T) ** 7) -
            ((SED_ref_rest < self.a_T) *  # Reference - Rest
             (1 - 7 * (SED_ref_rest / self.a_T) ** 2 +
              35 / 4 * (SED_ref_rest / self.a_T) ** 3 -
              7 / 2 * (SED_ref_rest / self.a_T) ** 5 +
              3 / 4 * (SED_ref_rest / self.a_T) ** 7)) -
            ((SED_rest_ref < self.a_T) *  # Rest - Reference
             (1 - 7 * (SED_rest_ref / self.a_T) ** 2 +
              35 / 4 * (SED_rest_ref / self.a_T) ** 3 -
              7 / 2 * (SED_rest_ref / self.a_T) ** 5 +
              3 / 4 * (SED_rest_ref / self.a_T) ** 7)) +
            ((SED_ref_ref < self.a_T) *  # Reference - References
             (1 - 7 * (SED_ref_ref / self.a_T) ** 2 +
              35 / 4 * (SED_ref_ref / self.a_T) ** 3 -
              7 / 2 * (SED_ref_ref / self.a_T) ** 5 +
              3 / 4 * (SED_ref_ref / self.a_T) ** 7)))) + 10e-9

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(SED_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                (h_u * h_v / SED_dips_dips ** 2) *
                ((
                 (SED_dips_dips < self.a_T) *  # first derivative
                 (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_dips / self.a_T ** 3 -
                                 35 / 2 * SED_dips_dips ** 3 / self.a_T ** 5 +
                                 21 / 4 * SED_dips_dips ** 5 / self.a_T ** 7))) +
                 (SED_dips_dips < self.a_T) *  # Second derivative
                 self.c_o_T * 7 * (9 * SED_dips_dips ** 5 - 20 * self.a_T ** 2 * SED_dips_dips ** 3 +
                                   15 * self.a_T ** 4 * SED_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
                (perpendicularity_matrix *
                 (SED_dips_dips < self.a_T) *  # first derivative
                 self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_dips / self.a_T ** 3 -
                               35 / 2 * SED_dips_dips ** 3 / self.a_T ** 5 +
                               21 / 4 * SED_dips_dips ** 5 / self.a_T ** 7)))
        )

        # Setting nugget effect of the gradients
        # TODO: This function can be substitued by simply adding the nugget effect to the diag
        C_G = T.fill_diagonal(C_G, -self.c_o_T * (-14 / self.a_T ** 2) + self.nugget_effect_grad_T)

        # Cross-Covariance gradients-interfaces
        C_GI = gi_reescale * (
            (hu_rest *
             (SED_dips_rest < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_rest / self.a_T ** 3 -
                              35 / 2 * SED_dips_rest ** 3 / self.a_T ** 5 +
                              21 / 4 * SED_dips_rest ** 5 / self.a_T ** 7))) -
            (hu_ref *
             (SED_dips_ref < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_ref / self.a_T ** 3 -
                              35 / 2 * SED_dips_ref ** 3 / self.a_T ** 5 +
                              21 / 4 * SED_dips_ref ** 5 / self.a_T ** 7)))
        ).T

        if self.u_grade_T.get_value() == 3:
            # ==========================
            # Condition of universality 1 degree

            # Gradients
            n = dips_position.shape[0]
            U_G = T.zeros((n * n_dimensions, n_dimensions))
            # x
            U_G = T.set_subtensor(
                U_G[:n, 0], 1)
            # y
            U_G = T.set_subtensor(
                U_G[n:n * 2, 1], 1
            )
            # z
            U_G = T.set_subtensor(
                U_G[n * 2: n * 3, 2], 1
            )

            # Interface
            U_I = -hx * gi_reescale

        elif self.u_grade_T.get_value() == 9:
            # ==========================
            # Condition of universality 2 degree
            # Gradients

            n = dips_position.shape[0]
            U_G = T.zeros((n * n_dimensions, 3 * n_dimensions))
            # x
            U_G = T.set_subtensor(U_G[:n, 0], 1)
            # y
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 1], 1)
            # z
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 2], 1)
            # x**2
            U_G = T.set_subtensor(U_G[:n, 3], 2 * gi_reescale * dips_position[:, 0])
            # y**2
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 4], 2 * gi_reescale * dips_position[:, 1])
            # z**2
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 5], 2 * gi_reescale * dips_position[:, 2])
            # xy
            U_G = T.set_subtensor(U_G[:n, 6], gi_reescale * dips_position[:, 1])           # This is y
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 6], gi_reescale * dips_position[:, 0])  # This is x
            # xz
            U_G = T.set_subtensor(U_G[:n, 7], gi_reescale * dips_position[:, 2])            # This is z
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 7], gi_reescale * dips_position[:, 0])  # This is x
            # yz
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 8], gi_reescale * dips_position[:, 2])   # This is z
            U_G = T.set_subtensor(U_G[n * 2:n * 3, 8], gi_reescale * dips_position[:, 1])   # This is y
           # Deprecated U_G = U_G

            # Interface
            U_I = - T.stack(
                gi_reescale * (rest_layer_points[:, 0] - ref_layer_points[:, 0]),
                gi_reescale * (rest_layer_points[:, 1] - ref_layer_points[:, 1]),
                gi_reescale * (rest_layer_points[:, 2] - ref_layer_points[:, 2]),
                gi_reescale ** 2 * (rest_layer_points[:, 0] ** 2 - ref_layer_points[:, 0] ** 2),
                gi_reescale ** 2 * (rest_layer_points[:, 1] ** 2 - ref_layer_points[:, 1] ** 2),
                gi_reescale ** 2 * (rest_layer_points[:, 2] ** 2 - ref_layer_points[:, 2] ** 2),
                gi_reescale ** 2 * (
                rest_layer_points[:, 0] * rest_layer_points[:, 1] - ref_layer_points[:, 0] * ref_layer_points[:, 1]),
                gi_reescale ** 2 * (
                rest_layer_points[:, 0] * rest_layer_points[:, 2] - ref_layer_points[:, 0] * ref_layer_points[:, 2]),
                gi_reescale ** 2 * (
                rest_layer_points[:, 1] * rest_layer_points[:, 2] - ref_layer_points[:, 1] * ref_layer_points[:, 2]),
            ).T

        # =================================
        # Creation of the Covariance Matrix
        # =================================
        C_matrix = T.zeros((length_of_C, length_of_C))

        # First row of matrices
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, 0:length_of_CG], C_G)

        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)

        if not self.u_grade_T.get_value() == 0:
            C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, -length_of_U_I:], U_G)

        # Second row of matrices
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG:length_of_CG + length_of_CGI], C_I)

        if not self.u_grade_T.get_value() == 0:
            C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, -length_of_U_I:], U_I)

            # Third row of matrices
            C_matrix = T.set_subtensor(C_matrix[-length_of_U_I:, 0:length_of_CG], U_G.T)
            C_matrix = T.set_subtensor(C_matrix[-length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], U_I.T)

        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module
        G_x = T.sin(T.deg2rad(dip_angles)) * T.sin(T.deg2rad(azimuth)) * polarity
        G_y = T.sin(T.deg2rad(dip_angles)) * T.cos(T.deg2rad(azimuth)) * polarity
        G_z = T.cos(T.deg2rad(dip_angles)) * polarity

        G = T.concatenate((G_x, G_y, G_z))

        # Creation of the Dual Kriging vector
        b = T.zeros_like(C_matrix[:, 0])
        b = T.set_subtensor(b[0:G.shape[0]], G)

        # Solving the kriging system
        # TODO: look for an eficient way to substitute nlianlg by a theano operation
        DK_parameters = T.dot(T.nlinalg.matrix_inverse(C_matrix), b)

        # ==============
        # Interpolator
        # ==============

        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        # Gradient contribution
        sigma_0_grad = T.sum(
            (weights[:length_of_CG, :] *
             gi_reescale *
             (-hu_SimPoint *
              (SED_dips_SimPoint < self.a_T) *  # first derivative
              (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * SED_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * SED_dips_SimPoint ** 5 / self.a_T ** 7)))),
            axis=0)

        # Interface contribution
        sigma_0_interf = (T.sum(
            -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
            (self.c_o_T * i_reescale * (
                (SED_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                (1 - 7 * (SED_rest_SimPoint / self.a_T) ** 2 +
                 35 / 4 * (SED_rest_SimPoint / self.a_T) ** 3 -
                 7 / 2 * (SED_rest_SimPoint / self.a_T) ** 5 +
                 3 / 4 * (SED_rest_SimPoint / self.a_T) ** 7) -
                ((SED_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                 (1 - 7 * (SED_ref_SimPoint / self.a_T) ** 2 +
                  35 / 4 * (SED_ref_SimPoint / self.a_T) ** 3 -
                  7 / 2 * (SED_ref_SimPoint / self.a_T) ** 5 +
                  3 / 4 * (SED_ref_SimPoint / self.a_T) ** 7)))), axis=0))

        # Universal drift contribution
        # Universal terms used to calculate f0
        _universal_terms_layers = T.horizontal_stack(
                rest_layer_points,
                (rest_layer_points ** 2),
                T.stack((rest_layer_points[:, 0] * rest_layer_points[:, 1],
                rest_layer_points[:, 0] * rest_layer_points[:, 2],
                rest_layer_points[:, 1] * rest_layer_points[:, 2]), axis=1)).T

        universal_matrix = T.horizontal_stack(
            (self.universal_matrix_T * yet_simulated).nonzero_values().reshape((9, -1)),
            _universal_terms_layers)

        if self.u_grade_T.get_value() == 0:
            f_0 = 0
        else:
            gi_rescale_aux = T.repeat(gi_reescale, 9)
            gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
            _aux_magic_term = T.tile(gi_rescale_aux[:grade_universal], (grid_val.shape[0], 1)).T
            f_0 = (T.sum(
                weights[-length_of_U_I:, :] * gi_reescale * _aux_magic_term *
                universal_matrix[:grade_universal]
                , axis=0))

        # Potential field
        # Value of the potential field
        Z_x = (sigma_0_grad + sigma_0_interf + f_0)[:-rest_layer_points.shape[0]]
        potential_field_interfaces = (sigma_0_grad + sigma_0_interf + f_0)[-rest_layer_points.shape[0]:]

        # Theano function to calculate a potential field
        self._interpolate = theano.function(
            [dips_position, dip_angles, azimuth, polarity, rest_layer_points, ref_layer_points,
             theano.In(yet_simulated, value=np.ones_like(self._grid.grid[:, 0]))],
            [Z_x, G_x, G_y, G_z, potential_field_interfaces, C_matrix, DK_parameters],
            on_unused_input="warn", profile=True, allow_input_downcast=True)

        #=======================================================================
        #               CODE TO EXPORT THE BLOCK DIRECTLY
        #========================================================================

        # Aux shared parameters
        infinite_pos = theano.shared(np.float32(np.inf))
        infinite_neg = theano.shared(np.float32(-np.inf))

        # Value of the lithology-segment
        n_formation = T.vector("The assigned number of the lithologies in this serie")

        # Loop to obtain the average Zx for every intertace
        def average_potential(dim_a, dim_b, pfi):
            """

            :param dim: size of the rest values vector per formation
            :param pfi: the values of all the rest values potentials
            :return: average of the potential per formation
            """
            average = pfi[T.cast(dim_a, "int32"): T.cast(dim_b, "int32")].sum()/(dim_b-dim_a)
            return average

        potential_field_unique, updates1 = theano.scan(fn=average_potential,
                                                       outputs_info=None,
                                                       sequences=dict(
                                                           input=T.concatenate((T.stack(0),
                                                                                n_formation,
                                                                                )), taps=[0, 1]),
                                                       non_sequences=potential_field_interfaces)

        # Loop to segment the distinct lithologies
        potential_field_iter = T.concatenate((T.stack(infinite_pos),
                                              potential_field_unique,
                                              T.stack(infinite_neg)))

        def compare(a, b, n_formation, Zx):
            return T.le(Zx, a) * T.ge(Zx, b) * n_formation

        block, updates2 = theano.scan(fn=compare,
                                      outputs_info=None,
                                      sequences=[dict(input=potential_field_iter, taps=[0, 1]),
                                                 n_formation],
                                      non_sequences=Z_x)

        # Adding to the block the contribution of the potential field
        potential_field_contribution = T.set_subtensor(
            self.block[T.nonzero(T.cast(yet_simulated, "int8"))[0]],
            block.sum(axis=0))

        # Some gradient testing
        # grad = T.jacobian(potential_field_contribution.sum(), rest_layer_points)
        # grad = T.grad(azimuth[0], potential_field_contribution)

        # Theano function to update the block
        self._block_export = theano.function([dips_position, dip_angles, azimuth, polarity, rest_layer_points,
                                             ref_layer_points, n_formation, yet_simulated], None,
                                            updates=[(self.block,  potential_field_contribution)],
                                            on_unused_input="warn", profile=True, allow_input_downcast=True)

















































