'''Create gravity and TMI plots
Created on Feb 12, 2015

@author: flow
'''
import os, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("/Users/flow/git/github/pygeomod/pygeomod")
sys.path.append("/Users/flow/git/pygeomod/pygeomod")
import pickle




if __name__ == '__main__':
    model_dir = r"/Users/flow/git/paper_sandstone/workdir/hres"
    os.chdir(model_dir)