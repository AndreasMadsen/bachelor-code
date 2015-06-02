
import warnings
import os.path as path
import sys
import os

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

import matplotlib as mpl
if (os.environ.get('DISPLAY') is None): mpl.use('Agg')
if (os.environ.get('BACKEND') is not None): mpl.use(os.environ.get('BACKEND'))

import theano

warnings.filterwarnings(
    action='error',
    category=UserWarning
)
warnings.filterwarnings(
    action='ignore',
    message='numpy.ndarray size changed, may indicate binary incompatibility'
)

if (os.environ.get('DTU_HPC') is not None):
    print('Running on HPC')

if (theano.config.optimizer == 'fast_run'):
    print('Theano optimizer enabled')
