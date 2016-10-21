'''Module with classes and methods to perform Bayesian Analyse in regional modelling.
Tested on Windows 8.1

Created on 02/12/2015

@author: Miguel de la Varga
'''

import numpy as np
# Geophysisc inversion
#import pynoddy

import subprocess
import os.path
import platform
# to create folder
import sys, os
import shutil
#import geobayes_simple as gs
from itertools import chain
 # PyMC 2 to perform Bayes Analysis
import pymc as pm
from pymc.Matplot import plot
from pymc import graph as gr
import pandas as pn
import numpy as np

import seaborn as sns; sns.set() # set default plot styles
from scipy.stats import kde
import pandas as pn
import pylab as P
from pymc.Matplot import plot

# Plotting setting
from IPython.core.pylabtools import figsize
from numpy.linalg import LinAlgError
figsize(12.5, 10)
# as we have our model and pygeomod in different paths, let's change the pygeomod path to the default path.
#sys.path.append("C:\Users\Miguel\workspace\pygeomod\pygeomod")

# Comunication with Geomodeller and grid management
import geogrid
# read out and change xml file
import geomodeller_xml_obj as gxml
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n\n\tMatplotlib not installed - plotting functions will not work!\n\n\n")
import colormaps as cmaps
plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='magma', cmap=cmaps.magma)
plt.set_cmap(cmaps.viridis)
class GeoPyMC_sim():
    "Object Definition to perform Bayes Analysis"

    def __init__(self, model_name):
        "nana"
        self.model_name = model_name

    def temp_creation(self,orig_dir,dest_dir = "temp/"):
        "Generates a working folder of the project"

        shutil.copytree(orig_dir, dest_dir)

        self.project_dir = dest_dir

    def proj_dir(self, proj_dir):
        self.project_dir = proj_dir+'\\'

    def read_excel(self, excl_dir, verbose = 0,  **kwds):
        "Reads data from an excel file"

        self.data = pn.read_excel(excl_dir)
        if verbose == 1:
            print "Excel Table: ",self.data
        self.data_ar = self.data.values[:,[0,1,2,3]]
        if kwds.has_key('columns'):
            self.data_ar = self.data.values[:, kwds['columns']]
        if verbose == 2:
            print "Obs_Id - Mean - Std - Type " ,self.data_ar

    # TO DO:
        # function to import a pandas table directly


    def set_interface_norm_distribution(self, **kwds):

        # Init contact point array
        self.contact_points_mc = []
        self.data_depth = np.asarray(self.data_ar[self.data_ar[:,3] == "Interface"][:,[0, 1,2]])

        # Set PyMC distribution per Stochastic contact point
        for i in range(len(self.data_depth)):
            self.contact_points_mc = np.append(self.contact_points_mc,
             pm.Normal(str(self.data_depth[i,0]), self.data_depth[i,1], 1./np.square(self.data_depth[i,2])))

    def set_azimuths_norm_distribution(self):

        # Init azimuths array
        self.azimuths_mc = []
        data_azimuth = np.asarray(self.data_ar[self.data_ar[:,3] == "Azimuth"][:,[0, 1,2]])
        # Set PyMC distribution per Stochastic azimuth
        for i in range(len(data_azimuth)):
            self.azimuths_mc = np.append(self.azimuths_mc,
            pm.Normal(str(data_azimuth[i,0]), data_azimuth[i,1], 1./np.square(data_azimuth[i,2])))

    def set_dips_norm_distribution(self):

        # Init dips array
        self.dips_mc = []
        data_dip = np.asarray(self.data_ar[self.data_ar[:,3] == "Dip"][:,[0, 1,2]])

        # Set PyMC distribution per Stochastic dip
        for i in range(len(data_dip)):
            self.dips_mc = np.append(self.dips_mc, pm.Normal(str(data_dip[i,0]), data_dip[i,1], 1./np.square(data_dip[i,2])))

    def set_Stoch_normal_distribution(self):
        self.set_interface_norm_distribution()
        self.set_azimuths_norm_distribution()
        self.set_dips_norm_distribution()


    def deterministic_GeoModel(self, xml_name, resolution = [50,50,50],
        noddy_geophy = False, densities = False, trace = False,
                                verbose = 0, **kwds):

        # In case that every interface is represented by two points in order to give horizonality
        two_points = kwds.get('two_points', False)

        plot_direction = kwds.get("plot_direction", "x")
        plot_cell = kwds.get("plot_cell", resolution[0]/2)
        z_dim = kwds.get("z_dim", False)

        # IMPORTANT NOTE: To be sure that the point we want to change fit with the Observation ID, I use the distribution name that
        # in this case is in contact_points(parent values). Children values (contact_points_val) only have the number itself

         # Create the array we will use to modify the xml. We have to check the order of the formations


        #==================================================
        # Loading old model
        #==============================================
        # Load the xml to be modify
        org_xml = self.project_dir+xml_name


        #Create the instance to modify the xml
            # Loading stuff

        gmod_obj = gxml.GeomodellerClass()
        gmod_obj.load_geomodeller_file(org_xml)


        if verbose > 1:
            print "Values original xml"
            gmod_obj.change_formation_values_PyMC(contact_points_mc = self.contact_points_mc,
                                                 azimuths_mc = self.azimuths_mc,
                                                 dips_mc = self.dips_mc,
                                                 info = True)
           #============================================
            # Modifing the model
            #===========================================
        gmod_obj.change_formation_values_PyMC(contact_points_mc = self.contact_points_mc,
                                             azimuths_mc = self.azimuths_mc,
                                             dips_mc = self.dips_mc,
                                             two_points = two_points)

            #==============================================
            # wtiting new model
            #============================================

        # Write the new xml

        gmod_obj.write_xml('temp\\temp_new.xml')

        # Read the new xml
        temp_xml = "temp\\temp_new.xml"
        G1 = geogrid.GeoGrid()

        # Getting dimensions and definning grid
        if noddy_geophy == False:
            G1.get_dimensions_from_geomodeller_xml_project(temp_xml)

        else:
            G1.xmin, G1.xmax   = np.min(self.ori_grav[:,0]), np.max(self.ori_grav[:,0])
            G1.ymin, G1.ymax    = np.min(self.ori_grav[:,1]), np.max(self.ori_grav[:,1])
            G1.zmin, G1.zmax    = z_dim[0], z_dim[1]

            G1.extent_x = G1.xmax - G1.xmin
            G1.extent_y = G1.ymax - G1.ymin
            G1.extent_z = G1.zmax - G1.zmin

        # Resolution!
        nx = resolution[0]
        ny = resolution[1]
        nz = resolution[2]
        G1.define_regular_grid(nx,ny,nz)

        # Updating project
        G1.update_from_geomodeller_project(temp_xml)

        if noddy_geophy == True:
            if densities == False:
                print "Provide a dictionary with the layer densities"
            else:
                densities = densities

            G1.analyse_geophysics(densities)

        if verbose > 0:
            G1.plot_section(plot_direction,cell_pos=plot_cell,colorbar = False, cmap = "coolwarm_r", fig_filename = "temp_xml.png"
                              , alpha = 1, figsize= (50,6),interpolation= 'nearest' ,
                               ve = 1, geomod_coord= True, contour = False)

            if noddy_geophy == "shit":
                print "Gravity Froward plot"
                plt.imshow(G1.geophys.grv_data, origin = "lower left",
                interpolation = 'nearest', cmap = 'viridis')

                plt.colorbar()

        if verbose > 1:
            print "Values changed xml"
            gmod_obj.change_formation_values_PyMC(contact_points_mc = self.contact_points_mc,
                                                 azimuths_mc = self.azimuths_mc,
                                                 dips_mc = self.dips_mc,
                                                 info = True)

        #self.model = G1
        return G1

    def creating_Bayes_model(self, constraints, verbose = 0):

        #CREATING THE MODEL

        # Chaining the Stochastic arrays
        parameters = list(chain(self.contact_points_mc,self.dips_mc,self.azimuths_mc))

        # Appending the rest
        for i in constraints:
            parameters = np.append(parameters, i)

        self.pymc_model = pm.Model(parameters)

        if verbose == 1:
            print self.pymc_model.variables


    def original_grav(self, ori_grav_obj, resolution = False, type = "xyz", verbose = 0, **kwds):
        """
        ** Arguments **
        type = xyz
            - *ori_grav* = string: path to the txt file
        type = grid
            - *ori_grav* = geogrid object: Geogrid with the gravity values
              (for example from a previous forward simulation)
        - *resolution* = vector: resolution (x,y)

        ** Keywords **

        - * Normalize * = bool: Normalize gravity between 0 and 1.
            """

        if type == "xyz":
            if resolution == False:
                raise AttributeError("A resolution is required for type 'xyz' gravity import" )
            self.ori_grav = np.loadtxt(ori_grav_obj)
            self.ori_grav_grid = self.ori_grav[:,3].reshape((resolution[1],resolution[0]))

        if type == "grid":
            self.ori_grav_grid = ori_grav_obj

        if kwds.has_key('Normalize'):
            self.ori_grav_grid = (self.ori_grav_grid-np.max(self.ori_grav_grid))/np.min(self.ori_grav_grid-np.max(self.ori_grav_grid))

        if verbose > 0:
            print "Gravity Contour plot"
            plt.imshow(self.ori_grav_grid, origin = "lower left",
            interpolation = 'nearest', cmap = 'jet')
            """
            plt.contourf(self.ori_grav_grid, cmap = 'gray' , alpha = 1, figsize= (12,12))
            plt.colorbar()
            """
    def MCMC_obj(self, db_name, path = "database_temp\\"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.Sim_MCMC = pm.MCMC(self.pymc_model,  db= "hdf5" , dbname= path+db_name+".hdf5")

    def dot_plot(self, path= "images_temp/",model_name = "PyMC_model",format = "png", **kwds):
        if not os.path.exists(path):
            os.makedirs(path)
        if format == "png":
            pm.graph.dag(self.pymc_model).write_png(path+model_name+"_dot_plot."+format)
        if format == "pdf":
            pm.graph.dag(self.pymc_model).write_pdf(path+model_name+"_dot_plot."+format)
        if format == "svg":
            pm.graph.dag(self.pymc_model).write_svg(path+model_name+"_dot_plot."+format)

        if kwds.has_key("display"):
            from IPython.core.display import Image
            return Image(path+model_name+"_dot_plot."+format)


            # TO DO: testing these two functions
    def set_priors(self, n_values = 1000):
        self.prior_dict = {}

        for points in self.contact_points_mc:
            try:
                self.prior_dict[points.__name__] = [points.random() for i in range(n_values)]
            except AttributeError:
                continue

        for dip in self.dips_mc:
            try:
                self.prior_dict[dip.__name__] = [dip.random() for i in range(n_values)]
            except AttributeError:
                continue

        for azimuth in self.azimuths_mc:
            try:
                self.prior_dict[azimuth.__name__] = [azimuth.random() for i in range(n_values)]
            except AttributeError:
                continue


    def export_priors_p(self, path = "prior.p"):
        import pickle
        pickle.dump(self.prior_dict, open( path, "wb" ))

class GeoPyMC_rep():
    def __init__(self, model_name):


        self.model_name = model_name

    def load_db(self, path, db_name, verbose = 0):
        self.LD = pm.database.hdf5.load(path+db_name)
        if verbose == 1:
            print "The number of chains of this data base are ",self.LD.chains

    def extract_GeoMods(self, trace_name = "model", n_samples = 9, burn = 0.2, n_chains = 1):
        if n_chains == "all":
            n_chains = self.LD.chains

        GeoMod_samples_all = []
        for i in range(1,n_chains+1):
            GeoMod_samples_all = np.append(GeoMod_samples_all,
            list(self.LD.trace(trace_name, chain = self.LD.chains - i )[:]))
        burnt = int(burn*len(GeoMod_samples_all))
        steps = np.linspace(burnt,len(GeoMod_samples_all)-1,n_samples, dtype = int)
        self.GeoMod_samples = GeoMod_samples_all[steps]

    def plot_lith_sect(self, section, cell_pos = "center",
     multiplots = False, n_plots = 9,  savefig = True, axes_style = "ticks",**kwds):

        colorbar = kwds.get('colorbar', True)
        cmap = kwds.get('cmap', 'jet')
        alpha = kwds.get('alpha', 1)
        rescale = kwds.get('rescale', False)
        ve = kwds.get('ve', 1.)
        figsize = kwds.get('figsize', (8,4))
        geomod_coord =  kwds.get('geomod_coord', False)
        contour = kwds.get('contour', False)
        linewidth = kwds.get("linewidth", 1)
        levels = kwds.get("plot_layer", None)
        path = kwds.get("path", "Plots")
        name = kwds.get("name", self.model_name+"_lith_sect")
        format = kwds.get("format", ".png")

        with sns.axes_style(axes_style):


            if multiplots == False:
                fig, axs = plt.subplots(1, 1, figsize=figsize)

                self.GeoMod_samples[-1].plot_section(section, cell_pos= cell_pos
                 ,colorbar = colorbar, ax = axs, alpha = alpha, cmap = cmap,
                                 interpolation= 'nearest' ,ve = ve, geomod_coord= geomod_coord,
                                  contour = contour, plot_layer = levels )

                plt.title( "Litholgies "+self.model_name,
                horizontalalignment='center',
                fontsize=20,)
                plt.tight_layout()
            if multiplots == True:
                n_rows = int(n_plots/3)
                fig, axs = plt.subplots(n_rows, 3, sharex=True, sharey=True, figsize = figsize)

                n_value = np.linspace(0, len(self.GeoMod_samples)-1, n_plots, dtype = int)

                plt.text(0.5, 1.1, "Litholgies "+self.model_name,
                horizontalalignment='center',
                fontsize=20,
                transform = axs[0,1].transAxes)

                for i, g  in enumerate(self.GeoMod_samples[n_value]):

                    g.plot_section(section, cell_pos= cell_pos
                     ,colorbar = colorbar, ax = axs[i- n_rows*(i/n_rows),i/n_rows], alpha = alpha, cmap = cmap,
                                     interpolation= 'nearest' ,ve = ve, geomod_coord= geomod_coord,
                                      contour = contour, plot_layer = levels )
                """
                plt.text(0.5, 1.1, "Litholgies",self.model_name,
                horizontalalignment='center',
                fontsize=20,
                transform = axs[0,1].transAxes)
                """

                plt.tight_layout()
            if savefig == True:
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path+name+format, transparent = True)


    def plot_grav_sect(self,
     multiplots = False, n_plots = 9, savefig = True, axes_style = "ticks",**kwds):

        colorbar = kwds.get('colorbar', True)
        cmap = kwds.get('cmap', 'jet')
        alpha = kwds.get('alpha', 1)
        rescale = kwds.get('rescale', False)
        ve = kwds.get('ve', 1.)
        figsize = kwds.get('figsize', (8,4))
        geomod_coord =  kwds.get('geomod_coord', False)
        contour = kwds.get('contour', False)
        linewidth = kwds.get("linewidth", 1)
        levels = kwds.get("plot_layer", None)
        path = kwds.get("path", "Plots")
        name = kwds.get("name", self.model_name+"_grav_sect")
        format = kwds.get("format", ".png")

        with sns.axes_style(axes_style):


            if multiplots == False:
                fig, ax = plt.subplots(1, 1, figsize=(13,13))


                ax.imshow(self.GeoMod_samples[-1].geophys.grv_data, cmap = cmap, origin = "lower")



                plt.title( "Gravity "+self.model_name,
                horizontalalignment='center',
                fontsize=20,)
                plt.tight_layout()

            if multiplots == True:
                n_rows = int(n_plots/3)
                n_value = np.linspace(0, len(self.GeoMod_samples)-1, n_plots, dtype = int)

                fig, axs = plt.subplots(n_rows, 3, sharex=True, sharey=True, figsize = (13, 13))

                for i, g  in enumerate(self.GeoMod_samples[n_value]):
                    axs[i- n_rows*(i/n_rows),i/n_rows].imshow(g.geophys.grv_data,
                    alpha = alpha, cmap = cmap,
                    interpolation= 'nearest', origin = "lower")

                plt.text(0.5, 1.1, "Gravity "+ self.model_name,
                horizontalalignment='center',
                fontsize=20,
                transform = axs[0,1].transAxes)

            if savefig == True:
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path+name+format, transparent = True)

    def calculate_prob_lith(self, n_samples = 100):
        import copy

        v_lith = np.unique(self.GeoMod_samples[-1].grid)
        self.prob_lith = np.zeros(len(v_lith), dtype = object)

        for i, pid in enumerate(v_lith):
            self.prob_lith[i] = copy.deepcopy(self.GeoMod_samples[i])
            self.prob_lith[i].grid = np.zeros_like(self.GeoMod_samples[i].grid)
            for lith in self.GeoMod_samples[-1:-n_samples:-1]:
                self.prob_lith[i].grid += (lith.grid == pid)/float(len(self.GeoMod_samples[-1:-n_samples:-1]))


    def calcualte_ie_masked(self):
        import copy
        h = copy.deepcopy(self.prob_lith[0])
        h.grid = np.zeros_like(h.grid)
        for layer in self.prob_lith:
            pm = np.ma.masked_equal(layer.grid, 0)
            h.grid -= (pm * np.ma.log2(pm)).filled(0)
        self.ie = h

    def total_ie(self, absolute = False):
        if not absolute == False:
            return np.sum(self.ie.grid)
        else:
            return np.sum(self.ie.grid)/np.size(self.ie.grid)

    # ==============================================================================

    def _select_trace(self,
        forbbiden = ["adaptive","model","deviance", "likelihood", "constrain",
        "Metropolis"] ):
        import copy
        # Init:
        self._plot_traces = copy.deepcopy(self.LD.trace_names[-1])
        iter = 0
        while iter<5:
            for i in self._plot_traces:
                if len(np.shape(self.LD.trace(i,chain = None)())) != 1:
                    self._plot_traces = np.delete(self._plot_traces, np.argwhere(self._plot_traces==i))
                    break
                for g in forbbiden:

                    if g in str(i):

                        self._plot_traces = np.delete(self._plot_traces, np.argwhere(self._plot_traces==i))
                        break
            iter += 1




    def import_prior_p(self, path = "priors.p"):
        import pickle
        self.prior_dict = pickle.load(open(path, "rb"))

    def plot_post(self, n_chains = None,  burn = 20., n_trace = 1000, savefig = True, **kwds):

        bin_size = kwds.get('bin_size', 25)
        path = kwds.get("path", "Plots/Posteriors/")
        name = kwds.get("name", self.model_name+"_posteriors")
        format = kwds.get("format", ".png")
        bins = []
        if not hasattr(self, "_plot_traces"):
            self._select_trace()

        n_values = np.linspace(len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])*(burn/100),
         len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])-1,n_trace, dtype = int)

        for Stoch in self._plot_traces:
            try:
                P.figure()
                x = self.LD.trace(Stoch, chain = None)[n_values]
                d = x
                n, bins, patches = P.hist(x, bin_size, normed=0.12, histtype='stepfilled', label = "Histogram", alpha = 0.1)
                P.setp(patches, 'facecolor', 'g', 'alpha', 0.5)

                try:
                    #prior
                    aux = np.linspace(float(np.min(self.prior_dict[Stoch])),float(np.max(self.prior_dict[Stoch])),15.)
                    bins = np.append(bins, aux)
                    bins = np.sort(bins)
                    mu = np.mean(self.prior_dict[Stoch])
                    sigma = np.std(self.prior_dict[Stoch])

                    # TO DO: Generalize this to not only normal distribution

                    y = P.normpdf( bins, mu, sigma)
                    l = P.plot(bins, y, 'k--', linewidth=1, label = "Prior distribution")
                except KeyError:
                    pass
                P.title("Posterior: "+ str(Stoch))
                P.ylabel("Prob")
                P.xlabel("Value")
                P.plot()
                try:
                    density = kde.gaussian_kde(d)
                    l = np.min(d)
                    u = np.max(d)
                    x = np.linspace(0, 1, 100) * (u - l) + l

                    P.plot(x, density(x), linewidth = 2, label = "Kernel Density Estimation", color = "r")

                except LinAlgError:
                    print Stoch
                    pass
                P.ylim([0, np.max(density(x))*1.4])
                P.legend(frameon = True)

                if savefig == True:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(path+name+"_"+Stoch+format, transparent = True)
            except AttributeError:
                "Trying to plot a no-numerical object: "+ Stoch
            except ValueError:
                "Some Stochastic has only one value" + Stoch
            except IndexError:
                Stoch
        return None


    def plot_joint3D(self, n_chains = None,  burn = 20., n_trace = 1000, axes_style = "ticks", savefig = True, **kwds):

        post3D =kwds.get("prior3D", "init")
        columns = kwds.get("columns", None)
        path = kwds.get("path", "Plots/")
        name = kwds.get("name", self.model_name+"_joint3D")
        format = kwds.get("format", ".png")
        bins = []
        if not hasattr(self, "_plot_traces"):
            self._select_trace()

        if len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])< n_trace:
            n_trace = len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])

        n_values = np.linspace(len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])*burn/100,
         len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])-1,n_trace, dtype = int)

        if not kwds.has_key("prior3D"):
            columns = []
            for i, Stoch in enumerate(self._plot_traces):
                if post3D == "init":
                    post3D = np.array(self.LD.trace(Stoch, chain = n_chains)[n_values])
                 #   print np.shape(post3D)
                    columns = np.append(columns,Stoch)
                else:
                    try:
                        post_aux = np.array(self.LD.trace(Stoch,chain = n_chains)[n_values])
                        post3D = np.column_stack((post3D, post_aux))
                        columns = np.append(columns,Stoch)
                    except KeyError:
                        pass
        df_3D = pn.DataFrame(post3D, columns = columns)
        with sns.axes_style(axes_style):
            grid = sns.PairGrid(df_3D, )
            grid.map_diag(plt.hist, bins=5, alpha=0.5 )
            grid.map_offdiag(sns.regplot,  color=".3")

            if savefig == True:
                if not os.path.exists(path):
                    os.makedirs(path)
                grid.savefig(path+name+"_"+Stoch+format, transparent = True)

    def plot_traces(self, n_chains = None,  burn = 20., n_trace = 1000, axes_style = "ticks", savefig = True, **kwds):

        post3D =kwds.get("prior3D", "init")
        columns = kwds.get("columns", None)
        path = kwds.get("path", "Plots/")
        name = kwds.get("name", self.model_name+"_traces")
        format = kwds.get("format", ".png")
        bins = []
        if not hasattr(self, "_plot_traces"):
            self._select_trace()

        if len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])< n_trace:
            n_trace = len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])

        n_values = np.linspace(len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])*burn/100,
         len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])-1,n_trace, dtype = int)
        if not kwds.has_key("prior3D"):
            columns = []
            for i, Stoch in enumerate(self._plot_traces):

                if post3D == "init":
                    post3D = np.array(self.LD.trace(Stoch, chain = n_chains)[n_values])
                    columns = np.append(columns,Stoch)
                else:
        #=======================================================================
                    try:
                        post_aux = np.array(self.LD.trace(Stoch,chain = n_chains)[n_values])
                        post3D = np.column_stack((post3D, post_aux))
                        columns = np.append(columns,Stoch)
                    except KeyError:
                        pass

        df = pn.DataFrame(post3D, columns = [columns])
        df.columns.name = 'Iterations'
        df.index.name = 'values'
        df.plot( subplots = True, layout=(-1, 2),figsize=(20, 20), linewidth = .7)

        if savefig == True:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path+name+"_"+Stoch+format, transparent = True)


    def _plot_geweke_f(self, data, name, format='png', suffix='-diagnostic', path='./', fontmap=None,
            verbose=1, axes_style = "ticks", **kwds):

        with sns.axes_style(axes_style):
            if not kwds.has_key('ax'):
                colorbar = kwds.get('colorbar', True)
                # create new axis for plot
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
            else:
                colorbar = False
                ax = kwds['ax']
            # Generate Geweke (1992) diagnostic plots
            if fontmap is None:
                fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

            # Generate new scatter plot

            x, y = np.transpose(data)
            ax.scatter(x.tolist(), y.tolist())

            # Plot options
            ax.set_xlabel('First iteration', fontsize='x-small')
            ax.set_ylabel('Z-score for %s' % name, fontsize='x-small')
            ax.set_title(name)

            # Plot lines at +/- 2 sd from zero
            ax.plot((np.min(x), np.max(x)), (2, 2), '--')
            ax.plot((np.min(x), np.max(x)), (-2, -2), '--')

            # Set plot bound
            ax.set_ylim(min(-2.5, np.min(y)), np.max(2.5, np.max(y)))
            ax.set_xlim(0, np.max(x))

    def plot_geweke(self, n_chains = None,  burn = 20., n_trace = 1000, axes_style = "ticks", savefig = True, **kwds):

        path = kwds.get("path", "Plots/")
        name = kwds.get("name", self.model_name+"_Geweke")
        format = kwds.get("format", ".png")

        Stochs = []
        if not hasattr(self, "_plot_traces"):
            self._select_trace()

        if len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])< n_trace:
            n_trace = len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])

        n_values = np.linspace(len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])*burn/100,
         len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])-1,n_trace, dtype = int)


        Stochs = self._plot_traces
        rows = int(len(Stochs)/3+1)
        fig, axs = plt.subplots(rows, 3, sharex=True, sharey=True, figsize = (13, 23))

        for i, Stoch in enumerate(Stochs):
            axa = axs[i/3,i- 3*(i/3)]
            try:
                geweke_val =  pm.geweke(self.LD.trace(Stoch, chain = n_chains)[n_values])
                self._plot_geweke_f(geweke_val, Stoch , ax = axa)
            except ValueError:
                print Stoch
                continue
            except LinAlgError:
                print "Alg",Stoch
                continue

            if savefig == True:
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path+name+"_"+Stoch+format, transparent = True)


    def plot_forest(self ,n_chains = None,  burn = 20., n_trace = 1000, axes_style = "ticks", savefig = True, **kwds):
        path = kwds.get("path", "Plots/")
        name = kwds.get("name", self.model_name+"_forest")
        format = kwds.get("format", ".png")
        post3D =kwds.get("prior3D", "init")
        columns = kwds.get("columns", None)

        self._select_trace(forbbiden = ["adaptive","model","deviance", "likelihood", "e_sq", "constrain",
        "Metropolis", "or"] )

        n_values = np.linspace(len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])*burn/100,
         len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])-1,n_trace, dtype = int)

        if not kwds.has_key("prior3D"):
            columns = []
            for i, Stoch in enumerate(self._plot_traces):

                if post3D == "init":
                    post3D = np.array(self.LD.trace(Stoch, chain = n_chains)[n_values])

                    columns = np.append(columns,Stoch)
                else:
        #=======================================================================
                    try:
                        post_aux = np.array(self.LD.trace(Stoch,chain = n_chains)[n_values])
                        post3D = np.column_stack((post3D, post_aux))
                        columns = np.append(columns,Stoch)
                    except KeyError:
                        pass

        df_3D = pn.DataFrame(post3D, columns = columns)
        df_3D.index.name = 'values'
        with sns.axes_style(axes_style):
            sns.boxplot(df_3D)
            plt.ylabel("Depth")
            plt.xlabel("Posterior Distribution")
            locs, labels = plt.xticks()
            plt.setp(labels, rotation=45)

            if savefig == True:
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path+name+"_"+Stoch+format, transparent = True)

class GeoPyMC_GeoMod_from_posterior(GeoPyMC_rep, GeoPyMC_sim):
    def __init__(self, db_name, path = "database_temp/",
        forbbiden = ["adaptive","model","deviance", "likelihood", "e_sq", "constrain",
        "Metropolis"]  ):
        #self.LD =
        self.load_db(path, db_name, verbose = 1)
        #self._plot_traces =
        self._select_trace(forbbiden = forbbiden)

    def recover_parameters(self, n_chains = None,  burn = 20., n_trace = 1000 ):
        Posterior = {}

        n_values = np.linspace(len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])*burn/100,
         len(self.LD.trace(self._plot_traces[0], chain = n_chains)[:])-1,n_trace,  dtype = int)
        b = []
        i = 0
        for i, Stoch in enumerate(self._plot_traces):
            if Stoch == "SM2_Atley" or Stoch == "BIF_Atley":
                continue
            def logp(value):
                return 0
            def random(Stoch = Stoch, i = i):
                b = self.LD.trace(Stoch, chain = n_chains)[n_values]
                i += 1
                return np.random.choice(b)

            Posterior[Stoch] = pm.Stochastic( logp = logp,
                        doc = 'na',
                        name = Stoch,
                        parents = {},
                        random = random,
                        trace = True,
                        value = 1900,
                        dtype=int,
                        rseed = 1.,
                        observed = False,
                        cache_depth = 2,
                        plot=True,
                        verbose = 0)

        self.contact_points_mc = []
        self.azimuths_mc = []
        self.dips_mc = []

        for pymc_obj in Posterior.iteritems():
            #print pymc_obj[0]
            if "Ori" in pymc_obj[0] and "_a" in pymc_obj[0]:
            #    print "iam ghere"
                self.azimuths_mc = np.append(self.azimuths_mc, pymc_obj[1])
            elif "Ori" in pymc_obj[0] and "_d" in pymc_obj[0]:
                self.dips_mc = np.append(self.dips_mc, pymc_obj[1])
            else:
                self.contact_points_mc = np.append(self.contact_points_mc, pymc_obj[1])
