
import warnings
import os.path as path
import sys
import os
import theano
import numpy as np

np.random.seed(42)

warnings.filterwarnings(
    action='error',
    category=UserWarning
)
warnings.filterwarnings(
    action='ignore',
    message='numpy.ndarray size changed, may indicate binary incompatibility'
)

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

if (os.environ.get('DTU_HPC') is not None):
    print('Running on HPC')

if (theano.config.optimizer == 'fast_run'):
    print('Theano optimizer enabled')
