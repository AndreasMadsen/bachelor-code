import os.path as path
import sys
import theano

thisdir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(thisdir, '..'))

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'warn'