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
#import matplotlib.pyplot as plt
from Visualization import GeoPlot


class Interpolator(GeoPlot):
    """
    Class which contain all needed methods to perform potential field implicit modelling
    """
    def __init__(self, range_var=None, c_o=None, nugget_effect=0.01, u_grade=9, rescaling_factor=1):
        """
        Basic interpolator parameters. Also here it is possible to change some flags of theano
        :param range_var: Range of the variogram, it is recommended the distance of the longest diagonal
        :param c_o: Sill of the variogram
        """
        # TODO: update Docstring

        theano.config.optimizer = "fast_run"
        theano.config.exception_verbosity = 'low'
        theano.config.compute_test_value = 'ignore'

        if not range_var:
            range_var = np.sqrt((self.xmax-self.xmin)**2 +
                                (self.ymax-self.ymin)**2 +
                                (self.zmax-self.zmin)**2)
        if not c_o:
            c_o = range_var**2/14/3

        # TODO: A function to update this values without calling the init again using .set_value
        self.a = theano.shared(range_var, "range", allow_downcast=True)
        self.c_o = theano.shared(c_o, "covariance at 0", allow_downcast=True)
        self.nugget_effect_grad = theano.shared(nugget_effect, "nugget effect of the grade", allow_downcast=True)
        self.u_grade = theano.shared(u_grade, allow_downcast=True)
        # TODO: To be sure what is the mathematical meaning of this

        self.rescaling_factor = theano.shared(rescaling_factor, "rescaling factor", allow_downcast=True)


        # TODO: Assert u_grade is 0,3 or 9

        # TODO: defne instances attribute in init using this parse arguments thing
    def create_regular_grid_2D(self):
        """
        Method to create a 2D regular grid where we interpolate
        :return: 2D regular grid for the resoulution nx, ny
        """
        try:
            g = np.meshgrid(
                np.linspace(self.xmin,self.xmax,self.nx, dtype="float32"),
                np.linspace(self.ymin,self.ymax,self.ny, dtype="float32"),
            )
            self.grid = np.vstack(map(np.ravel,g)).T.astype("float32")
        except AttributeError:
            print("Extent or resolution not provided. Use set_extent and/or set_resolutions first")

    def create_regular_grid_3D(self):
        """
        Method to create a 3D regurlar grid where is interpolated
        :return: 3D regurlar grid for the resolution nx,ny
        """
        try:
            g = np.meshgrid(
                np.linspace(self.xmin, self.xmax, self.nx, dtype="float32"),
                np.linspace(self.ymin, self.ymax, self.ny, dtype="float32"),
                np.linspace(self.zmin, self.zmax, self.nz, dtype="float32"), indexing="ij"
            )

            self.grid = np.vstack(map(np.ravel, g)).T.astype("float32")
            self._universal_matrix = np.vstack((
                self.grid.T,
                (self.grid ** 2).T,
                self.grid[:, 0] * self.grid[:, 1],
                self.grid[:, 0] * self.grid[:, 2],
                self.grid[:, 1] * self.grid[:, 2]))
        except AttributeError:
             raise AttributeError("Extent or resolution not provided. Use set_extent and/or set_resolutions first")

        # TODO Update shared value
        self.grid_val = theano.shared(self.grid, "Positions of the points to interpolate")
        self.universal_matrix = theano.shared(self._universal_matrix, "universal matrix")
    # TODO: Data management using pandas, find an easy way to add values

    # TODO: Once we have the data frame extract data to interpolator

    def load_data_csv(self, data_type, path=os.getcwd()):
        """
        Method to load either interface or foliations data csv files. Normally this is in which GeoModeller exports it
        :param data_type: string, 'interfaces' or 'foliations'
        :param path: path to the files
        :return: Pandas framework with the imported data
        """
        if data_type == "foliations":
            self.Foliations = pn.read_csv(path)
        else:
            self.Interfaces = pn.read_csv(path)


        try:
            getattr(self, "formations")
        except AttributeError:
            try:
                # Foliations may or may not be in all formations so we need to use Interfaces
                self.formations = self.Interfaces["formation"].unique()

                # TODO: Trying to make this more elegant?
                for el in self.formations:
                    for check in self.formations:
                        assert (el not in check or el == check), "One of the formations name contains other sting. Please rename."+str(el)+" in "+str(check)

                # TODO: Add the possibility to change the name in pandas directly (adding just a 1 in the contained string)

            except AttributeError:
                pass

            # TODO IMPORTANT: Decide if I calculate it once or not
        try:
            if self.rescaling_factor.get_value() == 1:
                max_coord = pn.concat([self.Foliations, self.Interfaces]).max()[:3]
                min_coord = pn.concat([self.Foliations, self.Interfaces]).min()[:3]

                self.rescaling_factor.set_value((np.max(max_coord - min_coord)))
        except AttributeError:
            pass

    def set_series(self, series_distribution=None):
        """
        The formations have to be separated by this thing! |
        :param series_distribution:
        :return:
        """
        if series_distribution == None:
            # TODO: Possibly we have to debug this function
            self.series = {"Default serie":self.formations}

        else:
            assert type(series_distribution) is dict, "series_distribution must be a dictionary, see Docstring for more information"
            assert sum(np.shape([i])[-1] for i in series_distribution.values()) is len(self.formations), "series_distribution must have the same number of values as number of formations %s." % self.formations
            self.series = series_distribution

    def compute_potential_field(self, series_name=0, verbose =0):
        """
        Compute potential field for the given serie
        :param series_name: name of the serie. Default first of the list
        :return: potential field of the serie
        """
        # I hate dictionaries
        if type(series_name) == int:
            if np.shape([list(self.series.values())[series_name]])[-1] > 1:
                serie = "|".join(list(self.series.values())[series_name])
            else:
                serie = list(self.series.values())[series_name]

        elif type(series_name) == str:
            if np.shape([self.series[series_name]])[-1] > 1:
                serie = "|".join(self.series[series_name])
            else:
                serie = self.series[series_name]

        self.dips_position = self.Foliations[self.Foliations["formation"].str.contains(serie)].as_matrix()[:, :3]
        self.dip_angles = self.Foliations[self.Foliations["formation"].str.contains(serie)]["dip"].as_matrix()
        self.azimuth = self.Foliations[self.Foliations["formation"].str.contains(serie)]["azimuth"].as_matrix()
        self.polarity = self.Foliations[self.Foliations["formation"].str.contains(serie)]["polarity"].as_matrix()



        if np.shape([self.series[series_name]])[-1]  == 1:
            print("I am in 1")
            self.layers = self.Interfaces[self.Interfaces["formation"].str.contains(serie)].as_matrix()[:, :-1]
            rest_layer_points = self.layers[1:]
            ref_layer_points = np.tile(self.layers[0], (np.shape(self.layers)[0]-1, 1))
        else:
            print("I am in 2")
            # TODO: This is ugly
            layers_list = []
            for formation in self.series[series_name]:
                layers_list.append(self.Interfaces[self.Interfaces["formation"] == formation].as_matrix()[:, :-1])
            self.layers = np.asarray(layers_list)
            rest_layer_points = np.vstack((i[1:] for i in self.layers))
            ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0]-1, 1)) for i in self.layers))

        if verbose > 0:
            print("The serie formations are %s" % serie)
            if verbose > 1:
                print("The formations are: \n"
                      "Layers ", self.Interfaces[self.Interfaces["formation"].str.contains(serie)], " \n "
                      "Foliations ", self.Foliations[self.Foliations["formation"].str.contains(serie)])

        self.Z_x, self.G_x, self.G_y, self.G_z, self.potential_interfaces, self.C, self.DK = self.interpolate(
            self.dips_position, self.dip_angles, self.azimuth, self.polarity,
            rest_layer_points, ref_layer_points)[:]

        self.potential_field = np.swapaxes(self.Z_x.reshape(self.nx, self.ny, self.nz),0,1)
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
        grade_universal = self.u_grade

        # Calculating the dimensions of the
        length_of_CG = dips_position.shape[0] * n_dimensions
        length_of_CGI = rest_layer_points.shape[0]
        length_of_U_I = grade_universal
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I

        # Extra parameters
        i_reescale = 1 / (self.rescaling_factor ** 2)
        gi_reescale = 1 / self.rescaling_factor

        # TODO: Check that the distances does not go nuts when I use too large numbers
        # ==========================================
        # Calculation of Cartesian and Euclidian distances
        # ===========================================
        # Auxiliary tile for dips and transformation to float 64 of variables in order to calculate precise euclidian
        # distances
        _aux_dips_pos = T.tile(dips_position, (n_dimensions, 1)).astype("float64")
        _aux_rest_layer_points = rest_layer_points.astype("float64")
        _aux_ref_layer_points = ref_layer_points.astype("float64")

        # This thing is the addition to simulate also the layer points
        grid_val = T.vertical_stack(self.grid_val, rest_layer_points)

        universal_terms_layers = T.horizontal_stack(
                rest_layer_points,
                (rest_layer_points ** 2),
                T.stack((rest_layer_points[:, 0] * rest_layer_points[:, 1],
                rest_layer_points[:, 0] * rest_layer_points[:, 2],
                rest_layer_points[:, 1] * rest_layer_points[:, 2]), axis=1)).T

        universal_matrix = T.horizontal_stack(self.universal_matrix, universal_terms_layers)


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
        C_I = (self.c_o * i_reescale * (
            (SED_rest_rest < self.a) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (SED_rest_rest / self.a) ** 2 +
             35 / 4 * (SED_rest_rest / self.a) ** 3 -
             7 / 2 * (SED_rest_rest / self.a) ** 5 +
             3 / 4 * (SED_rest_rest / self.a) ** 7) -
            ((SED_ref_rest < self.a) *  # Reference - Rest
             (1 - 7 * (SED_ref_rest / self.a) ** 2 +
              35 / 4 * (SED_ref_rest / self.a) ** 3 -
              7 / 2 * (SED_ref_rest / self.a) ** 5 +
              3 / 4 * (SED_ref_rest / self.a) ** 7)) -
            ((SED_rest_ref < self.a) *  # Rest - Reference
             (1 - 7 * (SED_rest_ref / self.a) ** 2 +
              35 / 4 * (SED_rest_ref / self.a) ** 3 -
              7 / 2 * (SED_rest_ref / self.a) ** 5 +
              3 / 4 * (SED_rest_ref / self.a) ** 7)) +
            ((SED_ref_ref < self.a) *  # Reference - References
             (1 - 7 * (SED_ref_ref / self.a) ** 2 +
              35 / 4 * (SED_ref_ref / self.a) ** 3 -
              7 / 2 * (SED_ref_ref / self.a) ** 5 +
              3 / 4 * (SED_ref_ref / self.a) ** 7)))) + 10e-9

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(SED_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                (h_u * h_v / SED_dips_dips ** 2) *
                ((
                     (SED_dips_dips < self.a) *  # first derivative
                     (-self.c_o * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_dips / self.a ** 3 -
                                   35 / 2 * SED_dips_dips ** 3 / self.a ** 5 + 21 / 4 * SED_dips_dips ** 5 / self.a ** 7))) +
                 (SED_dips_dips < self.a) *  # Second derivative
                 self.c_o * 7 * (9 * SED_dips_dips ** 5 - 20 * self.a ** 2 * SED_dips_dips ** 3 +
                                 15 * self.a ** 4 * SED_dips_dips - 4 * self.a ** 5) / (2 * self.a ** 7)) -
                (perpendicularity_matrix *
                 (SED_dips_dips < self.a) *  # first derivative
                 self.c_o * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_dips / self.a ** 3 -
                             35 / 2 * SED_dips_dips ** 3 / self.a ** 5 + 21 / 4 * SED_dips_dips ** 5 / self.a ** 7)))
        )

        # Setting nugget effect of the gradients
        C_G = T.fill_diagonal(C_G, -self.c_o * (-14 / self.a ** 2) + self.nugget_effect_grad)

        # Cross-Covariance gradients-interfaces
        C_GI = gi_reescale * (
            (hu_rest *
             (SED_dips_rest < self.a) *  # first derivative
             (- self.c_o * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_rest / self.a ** 3 -
                            35 / 2 * SED_dips_rest ** 3 / self.a ** 5 + 21 / 4 * SED_dips_rest ** 5 / self.a ** 7))) -
            (hu_ref *
             (SED_dips_ref < self.a) *  # first derivative
             (- self.c_o * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_ref / self.a ** 3 -
                            35 / 2 * SED_dips_ref ** 3 / self.a ** 5 + 21 / 4 * SED_dips_ref ** 5 / self.a ** 7)))
        ).T

        if self.u_grade.get_value() == 3:
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

        elif self.u_grade.get_value() == 9:
            # ==========================
            # Condition of universality 2 degree
            # Gradients

            n = dips_position.shape[0]
            U_G = T.zeros((n * n_dimensions, 3 * n_dimensions))
            # x
            U_G = T.set_subtensor(
                U_G[:n, 0], 1)
            # y
            U_G = T.set_subtensor(
                U_G[n * 1:n * 2, 1], 1
            )
            # z
            U_G = T.set_subtensor(
                U_G[n * 2: n * 3, 2], 1
            )
            # x**2
            U_G = T.set_subtensor(
                U_G[:n, 3], 2 * gi_reescale * dips_position[:, 0]
            )
            # y**2
            U_G = T.set_subtensor(
                U_G[n * 1:n * 2, 4], 2 * gi_reescale * dips_position[:, 1]
            )
            # z**2
            U_G = T.set_subtensor(
                U_G[n * 2: n * 3, 5], 2 * gi_reescale * dips_position[:, 2]
            )
            # xy
            U_G = T.set_subtensor(
                U_G[:n, 6], gi_reescale * dips_position[:, 1]  # This is y
            )

            U_G = T.set_subtensor(
                U_G[n * 1:n * 2, 6], gi_reescale * dips_position[:, 0]  # This is x
            )

            # xz
            U_G = T.set_subtensor(
                U_G[:n, 7], gi_reescale * dips_position[:, 2]  # This is z
            )
            U_G = T.set_subtensor(
                U_G[n * 2: n * 3, 7], gi_reescale * dips_position[:, 0]  # This is x
            )

            # yz

            U_G = T.set_subtensor(
                U_G[n * 1:n * 2, 8], gi_reescale * dips_position[:, 2]  # This is z
            )

            U_G = T.set_subtensor(
                U_G[n * 2:n * 3, 8], gi_reescale * dips_position[:, 1]  # This is y
            )

            U_G = U_G
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

        C_matrix = T.set_subtensor(
            C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)

        if not self.u_grade.get_value() == 0:
            C_matrix = T.set_subtensor(
                C_matrix[0:length_of_CG, -length_of_U_I:], U_G)

        # Second row of matrices
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, length_of_CG:length_of_CG + length_of_CGI], C_I)

        if not self.u_grade.get_value() == 0:
            C_matrix = T.set_subtensor(
                C_matrix[length_of_CG:length_of_CG + length_of_CGI, -length_of_U_I:], U_I)

            # Third row of matrices
            C_matrix = T.set_subtensor(
                C_matrix[-length_of_U_I:, 0:length_of_CG], U_G.T)
            C_matrix = T.set_subtensor(
                C_matrix[-length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], U_I.T)

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
              (SED_dips_SimPoint < self.a) *  # first derivative
              (- self.c_o * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_SimPoint / self.a ** 3 -
                             35 / 2 * SED_dips_SimPoint ** 3 / self.a ** 5 +
                             21 / 4 * SED_dips_SimPoint ** 5 / self.a ** 7)))),
            axis=0)

        # Interface contribution
        sigma_0_interf = (T.sum(
            -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
            (self.c_o * i_reescale * (
                (SED_rest_SimPoint < self.a) *  # SimPoint - Rest Covariances Matrix
                (1 - 7 * (SED_rest_SimPoint / self.a) ** 2 +
                 35 / 4 * (SED_rest_SimPoint / self.a) ** 3 -
                 7 / 2 * (SED_rest_SimPoint / self.a) ** 5 +
                 3 / 4 * (SED_rest_SimPoint / self.a) ** 7) -
                ((SED_ref_SimPoint < self.a) *  # SimPoint- Ref
                 (1 - 7 * (SED_ref_SimPoint / self.a) ** 2 +
                  35 / 4 * (SED_ref_SimPoint / self.a) ** 3 -
                  7 / 2 * (SED_ref_SimPoint / self.a) ** 5 +
                  3 / 4 * (SED_ref_SimPoint / self.a) ** 7)))), axis=0))

        # Potential field
        if self.u_grade.get_value() == 0:
            Z_x = (sigma_0_grad + sigma_0_interf)

        else:
            gi_rescale_aux = T.repeat(gi_reescale, 9)
            gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
            na = T.tile(gi_rescale_aux[:grade_universal], (grid_val.shape[0], 1)).T
            f_0 = (T.sum(
                weights[-length_of_U_I:, :] * gi_reescale * na *
                universal_matrix[:grade_universal], axis=0))

            Z_x = (sigma_0_grad + sigma_0_interf + f_0)[:-rest_layer_points.shape[0]]
            potential_field_interfaces = (sigma_0_grad + sigma_0_interf + f_0)[-rest_layer_points.shape[0]:]

        self.interpolate = theano.function(
            [dips_position, dip_angles, azimuth, polarity, rest_layer_points, ref_layer_points],
            [Z_x, G_x, G_y, G_z, potential_field_interfaces, C_matrix, DK_parameters],
            on_unused_input="warn", profile=True, allow_input_downcast=True)

    def theano_set_3D_nugget_degree0(self):
        dips_position = T.matrix("Position of the dips")
        dip_angles = T.vector("Angle of every dip")
        azimuth = T.vector("Azimuth")
        polarity = T.vector("Polarity")
        ref_layer_points = T.matrix("Reference points for every layer")
        rest_layer_points = T.matrix("Rest of the points of the layers")
        grid_val = theano.shared(self.grid, "Positions of the points to interpolate")
        universal_matrix = theano.shared(self._universal_matrix, "universal matrix")
        a = T.scalar()
        g = T.scalar()
        c = T.scalar()
        d = T.scalar()
        e = T.scalar("palier")
        f = T.scalar()
        # euclidean_distances = theano.shared(self.euclidean_distances, "list with all euclidean distances needed")

        # TODO: change all shared variables to self. in order to be able to change its value as well as check it. Othewise it will be always necesary to compile what makes no sense
        """
        SED_dips_dips = euclidean_distances[0]
        SED_dips_ref = euclidean_distances[1]
        SED_dips_rest = euclidean_distances[2]
        SED_dips_SimPoint = euclidean_distances[3]
        SED_ref_ref = euclidean_distances[4]
        SED_ref_rest = euclidean_distances[5]
        SED_ref_SimPoint = euclidean_distances[6]
        SED_rest_rest = euclidean_distances[7]
        SED_rest_ref = euclidean_distances[8]
        SED_rest_SimPoint = euclidean_distances[9]
        """

        # Init values

        n_dimensions = 3
        grade_universal = 9

        length_of_CG = dips_position.shape[0] * n_dimensions
        length_of_CGI = rest_layer_points.shape[0]
        length_of_U_I = grade_universal
        length_of_C = length_of_CG + length_of_CGI

        # ======
        # Intermediate steps for the calculation of the covariance function

        # Auxiliar tile for dips

        _aux_dips_pos = T.tile(dips_position, (n_dimensions, 1)).astype("float64")
        _aux_rest_layer_points = rest_layer_points.astype("float64")
        _aux_ref_layer_points = ref_layer_points.astype("float64")
        _aux_grid_val = grid_val.astype("float64")

        # Calculation of euclidian distances between the different elements


        SED_rest_rest = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_rest_layer_points.T))).astype("float64")

        SED_ref_rest = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_rest_layer_points.T))).astype("float64")

        SED_rest_ref = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_ref_layer_points.T))).astype("float64")

        SED_ref_ref = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_ref_layer_points.T))).astype("float64")

        SED_dips_dips = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_dips_pos ** 2).sum(1).reshape((1, _aux_dips_pos.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_dips_pos.T))).astype("float64")

        SED_dips_rest = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_rest_layer_points.T))).astype("float64")

        SED_dips_ref = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_ref_layer_points.T))).astype("float64")

        # Calculating euclidian distances between the point to simulate and the avalible data

        SED_dips_SimPoint = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_grid_val.T))).astype("float64")

        SED_rest_SimPoint = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_grid_val.T))).astype("float64")

        SED_ref_SimPoint = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_grid_val.T))).astype("float64")

        # Back to float64
        #     ref_layer_points = ref_layer_points.astype("float64")
        #  rest_layer_points = rest_layer_points.astype("float64")

        # =========
        # Cartesian distances

        # Cartesian distances between dips positions

        h_u = T.vertical_stack(
            T.tile(dips_position[:, 0] - dips_position[:, 0].reshape((dips_position[:, 0].shape[0], 1)), n_dimensions),
            # x
            T.tile(dips_position[:, 1] - dips_position[:, 1].reshape((dips_position[:, 1].shape[0], 1)), n_dimensions),
            # y
            T.tile(dips_position[:, 2] - dips_position[:, 2].reshape((dips_position[:, 2].shape[0], 1)),
                   n_dimensions))  # z

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

        # Perpendicularity matrix

        perpendicularity_matrix = T.zeros_like(SED_dips_dips)
        # 1D
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[0:dips_position.shape[0], 0:dips_position.shape[0]], 1)

        # 2D
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[dips_position.shape[0]:dips_position.shape[0] * 2,
            dips_position.shape[0]:dips_position.shape[0] * 2], 1)

        # 3D
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[dips_position.shape[0] * 2:dips_position.shape[0] * 3,
            dips_position.shape[0] * 2:dips_position.shape[0] * 3], 1)


        #if SED_rest_rest.shape[0] == 1:
      #  SED_rest_rest = T.tile(SED_rest_rest, (2,2))
      #  SED_ref_rest = T.tile(SED_ref_rest, (1,2))
      #  SED_rest_ref = T.tile(SED_rest_ref, (2,1))

        # ==================
        # Covariance matrix for interfaces

        C_I = a  *(
            (SED_rest_rest < self.a) * (1 - 7 * (SED_rest_rest / self.a) ** 2 +
                                        35 / 4 * (SED_rest_rest / self.a) ** 3 -
                                        7 / 2 * (SED_rest_rest / self.a) ** 5 +
                                        3 / 4 * (SED_rest_rest / self.a) ** 7) -
            (SED_ref_rest < self.a) * (1 - 7 * (SED_ref_rest / self.a) ** 2 +
                                       35 / 4 * (SED_ref_rest / self.a) ** 3 -
                                       7 / 2 * (SED_ref_rest / self.a) ** 5 +
                                       3 / 4 * (SED_ref_rest / self.a) ** 7) -
            (SED_rest_ref < self.a) * (1 - 7 * (SED_rest_ref / self.a) ** 2 +
                                       35 / 4 * (SED_rest_ref / self.a) ** 3 -
                                       7 / 2 * (SED_rest_ref / self.a) ** 5 +
                                       3 / 4 * (SED_rest_ref / self.a) ** 7) +
            (SED_ref_ref < self.a) * (1 - 7 * (SED_ref_ref / self.a) ** 2 +
                                      35 / 4 * (SED_ref_ref / self.a) ** 3 -
                                      7 / 2 * (SED_ref_ref / self.a) ** 5 +
                                      3 / 4 * (SED_ref_ref / self.a) ** 7)
        )


        i1 = (SED_rest_rest < self.a) * (1 - 7 * (SED_rest_rest / self.a) ** 2 +
                                        35 / 4 * (SED_rest_rest / self.a) ** 3 -
                                        7 / 2 * (SED_rest_rest / self.a) ** 5 +
                                        3 / 4 * (SED_rest_rest / self.a) ** 7)

        i2 = (SED_ref_rest < self.a) * (1 - 7 * (SED_ref_rest / self.a) ** 2 +
                                       35 / 4 * (SED_ref_rest / self.a) ** 3 -
                                       7 / 2 * (SED_ref_rest / self.a) ** 5 +
                                       3 / 4 * (SED_ref_rest / self.a) ** 7)

        i3 =  (SED_rest_ref < self.a) * (1 - 7 * (SED_rest_ref / self.a) ** 2 +
                                       35 / 4 * (SED_rest_ref / self.a) ** 3 -
                                       7 / 2 * (SED_rest_ref / self.a) ** 5 +
                                       3 / 4 * (SED_rest_ref / self.a) ** 7)

        i4 = (SED_ref_ref < self.a) * (1 - 7 * (SED_ref_ref / self.a) ** 2 +
                                      35 / 4 * (SED_ref_ref / self.a) ** 3 -
                                      7 / 2 * (SED_ref_ref / self.a) ** 5 +
                                      3 / 4 * (SED_ref_ref / self.a) ** 7)


        # =============
        # Covariance matrix for gradients at every xyz direction

        C_G = T.switch(
            T.eq(SED_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                (h_u * h_v / SED_dips_dips ** 2) *
                (((SED_dips_dips < self.a) *  # first derivative
                  (-f*((-14/self.a**2)+ 105/4*SED_dips_dips/self.a**3-
                 35/2*SED_dips_dips**3/self.a**5 + 21/4*SED_dips_dips**5/self.a**7))) +
                 (SED_dips_dips < self.a) *  # Second derivative
                f * 7*(9 * SED_dips_dips**5-20*self.a**2*SED_dips_dips**3+
                       15*self.a**4*SED_dips_dips-4*self.a**5)/(2*self.a**7)))
             - (perpendicularity_matrix *
                 (SED_dips_dips < self.a) *  # first derivative
                  f * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_dips / self.a ** 3 -
                  35 / 2 * SED_dips_dips ** 3 / self.a ** 5 + 21 / 4 * SED_dips_dips ** 5 / self.a ** 7)))



        C_G = T.fill_diagonal(C_G,  (-g * (-14 / self.a ** 2))+d)  # This sets the variance of the dips
        # ============
        # Cross-Covariance gradients-interfaces

        C_GI = e * ((
            (hu_rest) *
            (SED_dips_rest < self.a) *  # first derivative
            -f * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_rest / self.a ** 3 -
                 35 / 2 * SED_dips_rest ** 3 / self.a ** 5 + 21 / 4 * SED_dips_rest ** 5 / self.a ** 7)) -
            (hu_ref)  *
                (SED_dips_ref < self.a) *  # first derivative
                -f * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_ref / self.a ** 3 -
                     35 / 2 * SED_dips_ref ** 3 / self.a ** 5 + 21 / 4 * SED_dips_ref ** 5 / self.a ** 7)
        ).T

        gi1 =   ((hu_rest) *
            (SED_dips_rest < self.a) *  # first derivative
            f * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_rest / self.a ** 3 -
                 35 / 2 * SED_dips_rest ** 3 / self.a ** 5 + 21 / 4 * SED_dips_rest ** 5 / self.a ** 7))
        gi2 = ( (hu_ref)  *
                (SED_dips_ref < self.a) *  # first derivative
                f * ((-14 / self.a ** 2) + 105 / 4 * SED_dips_ref / self.a ** 3 -
                     35 / 2 * SED_dips_ref ** 3 / self.a ** 5 + 21 / 4 * SED_dips_ref ** 5 / self.a ** 7))

        # ===================
        # Creation of the Covariance Matrix

        C_matrix = T.zeros((length_of_C, length_of_C))

        # First row of matrices
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, 0:length_of_CG], C_G)

        C_matrix = T.set_subtensor(
            C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)


        # Second row of matrices
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, length_of_CG:length_of_CG + length_of_CGI], C_I)


        # =====================
        # Creation of the gradients G vector

        G_x = T.sin(T.deg2rad(dip_angles)) * T.sin(T.deg2rad(azimuth)) * polarity
        G_y = T.sin(T.deg2rad(dip_angles)) * T.cos(T.deg2rad(azimuth)) * polarity
        G_z = T.cos(T.deg2rad(dip_angles)) * polarity

        self.G_x = G_x
        self.G_y = G_y
        self.G_z = G_z

        G = T.concatenate((G_x, G_y, G_z))

        # ================
        # Creation of the kriging vector
        b = T.zeros_like(C_matrix[:, 0])
        b = T.set_subtensor(b[0:G.shape[0]], G)

        # ===============
        # Solving the kriging system

        DK_parameters = T.dot(T.nlinalg.matrix_inverse(C_matrix), b)

        # ==============
        # Interpolator
        # ==============


        # Cartesian distances between the point to simulate and the dips

        hu_SimPoint = T.vertical_stack(
            (dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
            (dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
            (dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        )

        weigths = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        # TODO multiply weights as a dot operation and not tiling it!
        sigma_0_grad = (
            T.sum(
                weigths[:length_of_CG, :] * e * hu_SimPoint / SED_dips_SimPoint * (
                    (SED_dips_SimPoint < self.a) * (  # first derivative
                        -7 * (self.a - SED_dips_SimPoint) ** 3 * SED_dips_SimPoint *
                        (8 * self.a ** 2 + 9 * self.a * SED_dips_SimPoint + 3 * SED_dips_SimPoint ** 2) * 1) /
                    (4 * self.a ** 7)
                ), axis=0))

        sigma_0_interf = (T.sum(
            weigths[length_of_CG:length_of_CG + length_of_CGI, :] * f*  # Covariance cubic to rest
            ((SED_rest_SimPoint < self.a) * (1 - 7 * (SED_rest_SimPoint / self.a) ** 2 +
                                             35 / 4 * (SED_rest_SimPoint / self.a) ** 3 -
                                             7 / 2 * (SED_rest_SimPoint / self.a) ** 5 +
                                             3 / 4 * (SED_rest_SimPoint / self.a) ** 7) -  # Covariance cubic to ref
             (SED_ref_SimPoint < self.a) * f * (1 - 7 * (SED_ref_SimPoint / self.a) ** 2 +
                                            35 / 4 * (SED_ref_SimPoint / self.a) ** 3 -
                                            7 / 2 * (SED_ref_SimPoint / self.a) ** 5 +
                                            3 / 4 * (SED_ref_SimPoint / self.a) ** 7)
             ), axis=0))


        Z_x = (sigma_0_grad + sigma_0_interf )
        """
        DK_parameters, C_matrix, SED_dips_rest,  hu_rest,
             hu_ref, i1,i2,i3,i4,
             gi1, gi2,
             C_G,
             G_x, G_y, G_z

        """


        self.geoMigueller = theano.function(
            [dips_position, dip_angles, azimuth, polarity, rest_layer_points, ref_layer_points, a, g, c, d, e, f],
            [Z_x, DK_parameters, C_matrix, SED_dips_rest,  hu_rest,
             hu_ref, i1,i2,i3,i4,
             gi1, gi2,
             C_G,
             G_x, G_y, G_z],
            on_unused_input="warn", profile=True, allow_input_downcast=True)

























































