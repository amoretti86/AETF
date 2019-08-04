from __future__ import print_function
import numpy as np
from AutoEncodingTopoFactors import AutoEncodingTopoFactor
from tfa_utils import plot, coordinate_tensor
from datetools import addDateTime
import sys
from hotspot_init import hot_spot_init
from TopoFactorAnalysisTF import TFA

from optparse import OptionParser
import pickle
import os


cur_date = addDateTime()
parser = OptionParser()

# Hyperparameters for the model
k = 25
#lr = 1e-5
#Epochs = 300
#batch_size = 50

parser.add_option("--rltdir", dest='rltdir', default='Experiment')
parser.add_option("--lr", dest='learning_rate', default=lr, type='float')
parser.add_option("--maxepochs", dest='maxepochs', default=Epochs, type='int')
parser.add_option("--batchsize", dest='batchsize', default=batch_size, type='int')


args = sys.argv
(options, args) = parser.parse_args(args)

#local_rlt_root = './rslts/NYU_Rest_30_(300)/'
local_rlt_root = './rslts/TFA/GRF_25/'
local_rlt_dir = local_rlt_root + options.rltdir + cur_date + '/'
RLT_DIR = local_rlt_dir

if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)


#subdir = '/Users/antoniomoretti/Documents/Research/peerlab/14.10.17_SpatialFactorAnalysis-master-3c97fd13aab276d6f7221dea986695ad6cb2e43b/'

subdir = os.path.join(os.getcwd(), 'data')
#'/Users/antoniomoretti/Documents/Research/peerlab/Spring18/AETF/data'

data = np.load(subdir + '/Sim03.npy')
train_data = data[0:5]
test_data = data[170:]

'''
train_data = np.load(subdir + "/NYU_Functional_AB.npy")
test_data = np.load(subdir + "/NYU_Functional_C.npy")

train_data = (train_data - np.mean(train_data)) / np.std(train_data)
test_data = (test_data - np.mean(test_data)) / np.std(test_data)

dim_data = list(train_data.shape[1:])
model = AutoEncodingTopoFactor(dim_data, k, options, RLT_DIR)
print("initialized")


scan_centers, scan_widths, scan_weights, test_error, test_centers, \
test_widths, test_weights, train_loss, train_mse = model.train(train_data, test_data)

#import pdb
#pdb.set_trace()


#total_error, train_mse = model.train_loss(train_data)



model_params = {
    'nsources' : k,
    'learning_rate' : lr,
    'epochs' : Epochs,
    'batch_size' : batch_size,
    'centers' : scan_centers,
    'widths'  : scan_widths,
    'weigts'  : scan_weights,
    'train_error' : train_loss,
    'train_mse' : train_mse,
    'test_error' : test_error,
    'test_centers' : test_centers,
    'test_widths' : test_widths,
    'test_weights' : test_weights,
    'Ytrain' : train_data,
    'Ytest' : test_data
}

pickle.dump( model_params, open(RLT_DIR + 'model_params.p', 'wb'))
'''

y = train_data

theta = {
    'width_log_pre': -1.0986122886681098, 
    'mu_var': np.array([177.03779675, 181.25900742]),
    'weight_log_pre': -0.6931471805599453, 
    'y_var': 0.01, 
    'center_log_pre': np.array([-7.47894834, -7.50251208]),
    'width_mean': 1.0,
    'center_mean': np.array([23.46954512, 21.94749435]),
    'weight_mean': 0.0
}

centers, widths = hot_spot_init(y, k)
print("Hotspot Init completed. Centers:\n", centers)
tfa = TFA(y, k, theta, center_init=centers, width_init=widths, save_dir=RLT_DIR)
print("fitting model 1...")
tfa.train()

#import pdb
#pdb.set_trace()



print('All done!')