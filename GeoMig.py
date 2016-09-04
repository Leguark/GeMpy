"""
Module with classes and methods to perform implicit regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 20/07 /2015

@author: Miguel de la Varga
"""

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class GeoPyMCSim:
    """Object Definition to perform Bayes Analysis"""

    def __init__(self, layers, name = "Kriging"):
        """nana"""
        self.model_name = name
        self.X = T.fmatrix('X')
        self.Y = T.fmatrix('Y')
        self.a = T.scalar('a')
        self.layers = layers
    # translation_vectors = X.reshape((X.shape[0], 1, -1)) - Y.reshape((1, Y.shape[0], -1))

    def euclidian_distances(self):
        squared_euclidean_distances = (self.X ** 2).sum(1).reshape((self.X.shape[0], 1)) + (self.Y ** 2)\
                                      .sum(1).reshape((1, self.Y.shape[0])) - 2 * self.X.dot(self.Y.T)
        self._euclidian_distances = theano.function([self.X, self.Y], T.sqrt(squared_euclidean_distances),  profile=False)

    def covariance_cubic(self, X,Y):
        r = T.fmatrix("distance")
        a = self.a
        # Theano function
        ans_d0 = (r < a) * (1 - 7 * (r / a) ** 2 + 35 / 4 *
                        (r / a) ** 3 - 7 / 2 * (r / a) ** 5
                        + 3 / 4 * (r / a) ** 7)
        self._covariance_cubic = theano.function([r,a], ans_d0,  profile=False, name = "predict")
        return self._covariance_cubic(self._euclidian_distances(X,Y),6)

    def covariance_cubic_d1(self):
        r = self._euclidian_distances
        a = self.a
        ans_d1 = [r < a] * (-7* (a - r)**3 *r* (8* a**2 + 9 *a* r + 3* r**2))/(4* a**7)
        return theano.function([self.euclidian_distances, self.a], ans_d1)

    def covariance_cubic_d2(self):
        r = self._euclidian_distances
        a = self.a
        ans_d2 = (-7 * (4. * a ** 5. - 15. * a ** 4. * r + 20. * (a ** 2) * (r ** 3) - 9 * r ** 5)
                 ) / (2 * a ** 7)
        return theano.function([self.euclidian_distances, self.a], ans_d2)

    def covariance_cubic_layers(self, verbose=0):
        """x = Array: Position of the measured points"
        a =  Range of the spherical semivariogram
        C_o = Nugget, variance
        """

        C_h = covariance_cubic()

        if verbose != 0:
            print(r_m > a)
            print("Our lag matrix is", self._euclidian_distances)
            print("Our covariance matrix is", C_h)
        return C_h

    def C_I(self):
        # print "layers", layers
       # layers = np.asarray(layers)
        # print "layers", len(layers)
        layers = self.layers
        self.euclidian_distances()

        for r in range(len(self.layers)):
            for s in range(len(self.layers)):
                #   print "layers2", layers[r][1:],layers[s][1:]

                # print "nagnoagjja", layers[s][0].reshape(1,-1),layers[r][1:],
                a_p = self.covariance_cubic(layers[r][1:], layers[s][1:])
                b_p = self.covariance_cubic(layers[s][0].reshape(1, -1),
                                      layers[r][1:], ).transpose()

                #test = cov_cubic_layer(layers[r][1:], layers[s][0].reshape(1, -1), a=a)
                c_p = self.covariance_cubic(layers[s][1:],
                                      layers[r][0].reshape(1, -1), ).transpose()
                d_p = self.covariance_cubic(layers[r][0].reshape(1, -1),
                                      layers[s][0].reshape(1, -1),)

                # pdb.set_trace()
                # print "s", s,"r", r
                if s == 0:

                    C_I_row = a_p - b_p - c_p + d_p
                else:
                    C_I_row = np.hstack((C_I_row, a_p - b_p - c_p + d_p))

            if r == 0:
                C_I = C_I_row
            else:
                C_I = np.vstack((C_I, C_I_row))
                # C_I  += 0.00000001
        return C_I


class GeoMigSim_pro2:
    def __init__(self, range = 6, c_o = -0.58888888):
        """general thigs"""
        theano.config.optimizer = "fast_run"
        theano.config.exception_verbosity = 'low'
        theano.config.compute_test_value = 'off'


        self.a = theano.shared(range, "range")
        self.c_o = theano.shared(c_o, "covariance at 0")

    def theano_set(self,  range = 6, c_o = -0.58888888):


        #self.n_layers = len(layers_n)
        self.dips = T.fmatrix("dips")
        self.layers = T.fmatrix('position of the layers')
        self.X = T.fmatrix('X')
        self.Y = T.fmatrix('Y')


        self.V1 = T.fmatrix('V1')
        self.V2 = T.fmatrix('V2')
        self.layers = T.fmatrix("layers")
        perpendicularity_matrix = T.bmatrix

        _squared_euclidean_distances = T.sqrt((self.X ** 2).sum(1).reshape((self.X.shape[0], 1)) + (self.Y ** 2).sum(1).reshape((1, self.Y.shape[0])) - 2 * self.X.dot(self.Y.T))


        # evaluation of the covoraiance
        r = _squared_euclidean_distances
        self._ans_d0 = (r < self.a) * (1 - 7 * (r / self.a) ** 2 + 35 / 4 *
                                       (r / self.a) ** 3 - 7 / 2 * (r / self.a) ** 5
                                       + 3 / 4 * (r / self.a) ** 7)


        self._ans_d1 = (r < self.a) * (-7 * (self.a - r) ** 3 * r * (8 * self.a ** 2 + 9 * self.a * r +                         3 * r ** 2) * 1) / (4 * self.a ** 7)

        self._ans_d2 = (r < self.a) * (-7 * (4. * self.a ** 5. - 15. * self.a ** 4. * r + 20. * (self.a                         ** 2) * (r ** 3) - 9 * r ** 5) * 1) / (2 * self.a ** 7)

        #==========================
        # CI
        #==========================
        self.f_cov = theano.function([self.X,self.Y], (self._ans_d0), profile=False, name="predict")
        #==========================
        # Gradient covariance
        #==========================

        _squared_euclidean_distances_1D_a = self.V1 - self.V1.reshape((self.V1.shape[0],1))
        _squared_euclidean_distances_1D_b = self.V2 - self.V2.reshape((self.V2.shape[0],1))

        h1 = T.vertical_stack(
                     T.tile(self.dips[:,0] - self.dips[:,0].reshape((self.dips[:,0].shape[0],1)),2),
                     T.tile(self.dips[:,1] - self.dips[:,1].reshape((self.dips[:,1].shape[0],1)),2),)
                     #T.tile(self.dips[:,2] - self.dips[:,2].reshape((self.dips[:,2].shape[0],1)),3))

        # perpendicularity_matrix =theano.reduce(
        #             fn = lambda x,M: T.fill_diagonal_offset(M,1,x),
        #             sequences =  T.arange(-self.dips.shape[1],self.dips.shape[1]+1),
        #             outputs_info = [T.zeros_like(r)]
        # )[0]

        perpendicularity_matrix = T.zeros_like(r)
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[:self.dips.shape[1],:self.dips.shape[1]], 1 )

        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips.shape[1]:self.dips.shape[1]*2,
            self.dips.shape[1]:self.dips.shape[1]*2], 1)

    #    perpendicularity_matrix = T.set_subtensor(
    #        perpendicularity_matrix[self.dips.shape[1]*2:self.dips.shape[1]*3,
     #       self.dips.shape[1]*2:self.dips.shape[1] * 3 ], 1)



        h2 = -h1.T
        a = self.c_o*(-h1*h2/r**2)* ((1/r)*self._ans_d1-self._ans_d2)
        b = perpendicularity_matrix*self._ans_d1/r

        self._ans_grad = T.switch(T.eq(r,0), -0.58,(
            (self.c_o*(((h1*h2/r**2)* ((1/r)*self._ans_d1-self._ans_d2)))+
             self.c_o *perpendicularity_matrix*self._ans_d1/r)))


        self.f_CG = theano.function([self.dips, self.X,self.Y], (self._ans_grad, perpendicularity_matrix ))

        #theano.printing.pydotprint(self._ans_grad )


    def theano_set2(self):
        aux_X = T.dmatrix("aux_X")
        aux_Y = T.dmatrix("aux Y")
        dips = T.dmatrix("dips")


    def call_theano_for_CI(self, layers):

        auxi = np.vstack((i[1:] for i in layers))
        auxi2 = np.vstack((np.tile(i[0],(np.shape(i)[0]-1,1)) for i in layers))
        a_p_matrix = self.f_cov(auxi,auxi)
        b_p_matrix = self.f_cov(auxi2,auxi)
        c_p_matrix = self.f_cov(auxi,auxi2)
        d_p_matrix = self.f_cov(auxi2,auxi2)
        self._C_I = a_p_matrix-b_p_matrix-c_p_matrix+d_p_matrix


        return self._C_I

    def call_theano_for_CG(self, dips):
        dips = np.vstack((i for i in dips))
        aux_x =  np.tile(dips, (2,1))
      #  aux_v = np.vstack((np.tile(dips[:,0],(3,1)),np.tile(dips[:,1],(3,1)),np.tile(dips[:,2],(3,1))))
        self._CG = self.f_CG(dips, aux_x, aux_x)
        return self._CG



    def call_theano_for_C_GI(self,dips, layers):
        dips = np.vstack((i for i in dips))
        aux_X = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))
        aux_Y = np.vstack((i[1:] for i in layers))
        self._CGI = self._f_CGI(dips, aux_X,aux_Y)
        return self._CGI



    def create_regular_grid(self, x_min,x_max, y_min, y_max, nx,ny):


        g = np.meshgrid(
            np.linspace(x_min,x_max,nx, dtype="float32"),
            np.linspace(y_min,y_max,ny, dtype="float32"),
        )

        self.grid = np.vstack(map(np.ravel,g)).T.astype("float32")


    def theano_set3(self):

        dips_position = T.matrix("Position of the dips")
        dip_angles = T.vector("Angle of every dip")
        ref_layer_points = T.matrix("Reference points for every layer")
        rest_layer_points = T.matrix("Rest of the points of the layers")
        grid_val = theano.shared(self.grid, "Positions of the points to interpolate")

        # Init values

        n_dimensions = 2
        grade_universal = 2

        length_of_CG = dips_position.shape[0] * n_dimensions
        length_of_CGI = rest_layer_points.shape[0]
        length_of_U_I = grade_universal
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I


        # ======
        # Intermediate steps for the calculation of the covariance function

        # Auxiliar tile for dips

        _aux_dips_pos = T.tile(dips_position, (n_dimensions,1))

        # Calculation of euclidian distances between the different elements

        SED_rest_rest = T.sqrt(
            (rest_layer_points ** 2).sum(1).reshape((rest_layer_points.shape[0], 1)) +
            (rest_layer_points ** 2).sum(1).reshape((1, rest_layer_points.shape[0])) -
            2 * rest_layer_points.dot(rest_layer_points.T))

        SED_ref_rest = T.sqrt(
            (ref_layer_points ** 2).sum(1).reshape((ref_layer_points.shape[0], 1)) +
            (rest_layer_points ** 2).sum(1).reshape((1, rest_layer_points.shape[0])) -
            2 * ref_layer_points.dot(rest_layer_points.T))


        SED_rest_ref = T.sqrt(
            (rest_layer_points ** 2).sum(1).reshape((rest_layer_points.shape[0], 1)) +
            (ref_layer_points ** 2).sum(1).reshape((1, ref_layer_points.shape[0])) -
            2 * rest_layer_points.dot(ref_layer_points.T))

        SED_ref_ref = T.sqrt(
            (ref_layer_points ** 2).sum(1).reshape((ref_layer_points.shape[0], 1)) +
            (ref_layer_points ** 2).sum(1).reshape((1, ref_layer_points.shape[0])) -
            2 * ref_layer_points.dot(ref_layer_points.T))

        SED_dips_dips = T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_dips_pos ** 2).sum(1).reshape((1, _aux_dips_pos.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_dips_pos.T))

        SED_dips_rest = T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (rest_layer_points ** 2).sum(1).reshape((1, rest_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(rest_layer_points.T))

        SED_dips_ref = T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (ref_layer_points ** 2).sum(1).reshape((1, ref_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(ref_layer_points.T))

        # Cartesian distances between dips positions

        h_u = T.vertical_stack(
            T.tile(dips_position[:, 0] - dips_position[:, 0].reshape((dips_position[:, 0].shape[0], 1)), 2),   #x
            T.tile(dips_position[:, 1] - dips_position[:, 1].reshape((dips_position[:, 1].shape[0], 1)), 2), ) #y
        # T.tile(self.dips[:,2] - self.dips[:,2].reshape((self.dips[:,2].shape[0],1)),3))          #z

        h_v = h_u.T

        # Cartesian distances between dips and interface points

        hu_rest = T.vertical_stack(
        (dips_position[:, 0] - rest_layer_points[:, 0].reshape((rest_layer_points[:, 0].shape[0], 1))).T,
        (dips_position[:, 1] - rest_layer_points[:, 1].reshape((rest_layer_points[:, 1].shape[0], 1))).T
        )

        hu_ref = T.vertical_stack(
        (dips_position[:, 0] - ref_layer_points[:, 0].reshape((ref_layer_points[:, 0].shape[0], 1))).T,
        (dips_position[:, 1] - ref_layer_points[:, 1].reshape((ref_layer_points[:, 1].shape[0], 1))).T)

        # Cartesian distances between reference points and rest

        hx = T.stack(
        (rest_layer_points[:, 0] - ref_layer_points[:, 0]),
        (rest_layer_points[:, 1] - ref_layer_points[:, 1])
        ).T

        # Perpendicularity matrix

        perpendicularity_matrix = T.zeros_like(SED_dips_dips)
        # 1D
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[dips_position.shape[1], :dips_position.shape[1]], 1)

        # 2D
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[dips_position.shape[1]:dips_position.shape[1] * 2,
            dips_position.shape[1]:dips_position.shape[1] * 2], 1)

        # 3D
        #    perpendicularity_matrix = T.set_subtensor(
        #        perpendicularity_matrix[self.dips.shape[1]*2:self.dips.shape[1]*3,
        #       self.dips.shape[1]*2:self.dips.shape[1] * 3 ], 1)

        # ==================
        # Covariance matrix for interfaces

        C_I = (
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

        # =============
        # Covariance matrix for gradients at every xyz direction

        C_G = T.switch(
                       T.eq(SED_dips_dips, 0), # This is the condition
                       0,                     # If true it is equal to 0. This is how a direction affect another
                       (                      # else
                        #self.c_o*
                         (-h_u*h_v/SED_dips_dips**2)* ((1/SED_dips_dips) *
                         (SED_dips_dips < self.a) * (                                # first derivative
                         -7 * (self.a - SED_dips_dips) ** 3 * SED_dips_dips *
                         (8 * self.a ** 2 + 9 * self.a * SED_dips_dips + 3 * SED_dips_dips ** 2) * 1) /
                         (4 * self.a ** 7) -
                         (SED_dips_dips < self.a) * (                                # Second derivative
                         -7 * (4. * self.a ** 5. - 15. * self.a ** 4. * SED_dips_dips + 20. *
                         (self.a ** 2) * (SED_dips_dips ** 3) - 9 * SED_dips_dips ** 5) * 1) /
                         (2 * self.a ** 7)) +
                        # self.c_o *
                         perpendicularity_matrix *
                        (SED_dips_dips < self.a) * (                                 # first derivative
                         -7 * (self.a - SED_dips_dips) ** 3 * SED_dips_dips *
                         (8 * self.a ** 2 + 9 * self.a * SED_dips_dips + 3 * SED_dips_dips ** 2) * 1) /
                         (4 * self.a ** 7)
                        )
                       )
        C_G = T.fill_diagonal(C_G, self.c_o) # This sets the variance of the dips

        # ============
        # Cross-Covariance gradients-interfaces

        C_GI =(
            hu_rest / SED_dips_rest *
            (SED_dips_rest < self.a) * (       # first derivative
            -7 * (self.a - SED_dips_rest) ** 3 * SED_dips_rest *
            (8 * self.a ** 2 + 9 * self.a * SED_dips_rest + 3 * SED_dips_rest ** 2) * 1) /
            (4 * self.a ** 7) -
            hu_ref / SED_dips_ref *
            (SED_dips_ref < self.a) * (  # first derivative
            -7 * (self.a - SED_dips_ref) ** 3 * SED_dips_ref *
            (8 * self.a ** 2 + 9 * self.a * SED_dips_ref + 3 * SED_dips_ref ** 2) * 1) /
            (4 * self.a ** 7)
        ).T

        # ==========================
        # Condition of universality
        # Gradients

        n = dips_position.shape[0]
        U_G = T.zeros((n*2,2))
        U_G = T.set_subtensor(
            U_G[:n,0], 1)
        U_G = T.set_subtensor(
            U_G[n:,1],1
        )

        # Interface
        U_I = hx


        # ===================
        # Creation of the Covariance Matrix



        C_matrix = T.zeros((length_of_C, length_of_C))

        # First row of matrices
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG , 0:length_of_CG], C_G)

        C_matrix = T.set_subtensor(
            C_matrix[0:length_of_CG , length_of_CG:length_of_CG  + length_of_CGI], C_GI.T)

        C_matrix = T.set_subtensor(
            C_matrix[0:length_of_CG, -length_of_U_I:], U_G)

        # Second row of matrices
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG] ,C_GI)
        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, length_of_CG:length_of_CG + length_of_CGI] ,C_I)

        C_matrix = T.set_subtensor(
            C_matrix[length_of_CG:length_of_CG + length_of_CGI, -length_of_U_I:], U_I)


        # Third row of matrices
        C_matrix = T.set_subtensor(
            C_matrix[-length_of_U_I:, 0:length_of_CG], U_G.T)
        C_matrix = T.set_subtensor(
            C_matrix[-length_of_U_I:, length_of_CG:length_of_CG+length_of_CGI] ,U_I.T)

        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module:

        arrow_point_positions_x =  T.cos(T.deg2rad(dip_angles))
        arrow_point_positions_y =  T.sin(T.deg2rad(dip_angles))
        arrow_point_position = T.concatenate((arrow_point_positions_x, arrow_point_positions_y))

        G = arrow_point_position

        # ================
        # Creation of the kriging vector
        b = T.zeros_like(C_matrix[:, 0])
        b = T.set_subtensor(b[0:G.shape[0]],G)



        # ===============
        # Solving the kriging system

        DK_parameters= T.dot(T.nlinalg.matrix_inverse(C_matrix),b)


        # ==============
        # Interpolator
        # ==============
        # Calculating euclidian distances between the point to simulate and the avalible data

        SED_dips_SimPoint = T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (grid_val ** 2).sum(1).reshape((1, grid_val.shape[0])) -
            2 * _aux_dips_pos.dot(grid_val.T))

        SED_rest_SimPoint = T.sqrt(
            (rest_layer_points ** 2).sum(1).reshape((rest_layer_points.shape[0], 1)) +
            (grid_val ** 2).sum(1).reshape((1, grid_val.shape[0])) -
            2 * rest_layer_points.dot(grid_val.T))

        SED_ref_SimPoint = T.sqrt(
            (ref_layer_points ** 2).sum(1).reshape((ref_layer_points.shape[0], 1)) +
            (grid_val ** 2).sum(1).reshape((1, grid_val.shape[0])) -
            2 * ref_layer_points.dot(grid_val.T))

        # Cartesian distances between the point to simulate and the dips

        hu_SimPoint = T.vertical_stack(
        (dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
        (dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T
        )


        weigths = T.tile(DK_parameters, (grid_val.shape[0],1)).T




        sigma_0_grad = (T.sum(
            weigths[:length_of_CG, :] * hu_SimPoint/SED_dips_SimPoint
            * (
            (SED_dips_SimPoint < self.a) * (  # first derivative
                -7 * (self.a - SED_dips_SimPoint) ** 3 * SED_dips_SimPoint *
                (8 * self.a ** 2 + 9 * self.a * SED_dips_SimPoint + 3 * SED_dips_SimPoint ** 2) * 1) /
            (4 * self.a ** 7)
        ), axis = 0))


        sigma_0_interf = (T.sum(
            weigths[length_of_CG:length_of_CG+length_of_CGI, :]*    # Covariance cubic to rest
                ((SED_rest_SimPoint < self.a) * (1 - 7 * (SED_rest_SimPoint / self.a) ** 2 +
            35 / 4 * (SED_rest_SimPoint / self.a) ** 3 -
            7 / 2 * (SED_rest_SimPoint / self.a) ** 5 +
            3 / 4 * (SED_rest_SimPoint / self.a) ** 7) -          # Covariance cubic to ref
            (SED_ref_SimPoint < self.a) * (1 - 7 * (SED_ref_SimPoint / self.a) ** 2 +
            35 / 4 * (SED_ref_SimPoint / self.a) ** 3 -
            7 / 2 * (SED_ref_SimPoint / self.a) ** 5 +
            3 / 4 * (SED_ref_SimPoint / self.a) ** 7)
                 ), axis = 0))

        f_0 = grid_val.T
        f_0 = (T.sum(
          weigths[-length_of_U_I:,:]* grid_val.T, axis = 0))


        Z_x = sigma_0_grad + sigma_0_interf + f_0
        """
        """
        self.geoMigueller = theano.function([dips_position, dip_angles, rest_layer_points, ref_layer_points], [Z_x,
                                C_I,C_G,C_GI,weigths], on_unused_input="warn", profile= True)





