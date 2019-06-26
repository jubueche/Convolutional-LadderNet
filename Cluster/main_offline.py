from ln_cnn_offline import *

import random
from timeit import default_timer as timer
import csv
from hyperopt import STATUS_OK, STATUS_FAIL
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import numpy as np
from hyperopt import Trials
from hyperopt import tpe
from hyperopt import fmin

import sys

# Number of rounds used for confidence interval. Use 5.
num_trials = 5

# Max. number of search iterations for Hyperopt. Use 100.
MAX_EVALS = 500
SAVED = False

# Set to False in order to use full data set
DEBUG = False

# Number of epochs used
NUM_EPOCHS = 10

# Total number of labeled points. Divide by 9 to get num. per class for PaviaU dataset.
NUM_LABELED = 90


def convert_params(params):
      dic = {}
      for k,v in params.items():
            
            if(isinstance(v, tuple)):
                  s = ['{:.3f}'.format(vv) for vv in v]
                  dic[k] = s
            else:
                  dic[k] = '{:.3f}'.format(v)
      return dic

def objective(params):

  global ITERATION  
  ITERATION += 1
  status = STATUS_OK

  start = timer()

  X_train, y_train, X_test, y_test = get_pavia(debug = DEBUG, numComponents=30,windowSize=7,saved=SAVED)
  accuracies = []
  np.random.seed(12)
  for i in range(num_trials):
    #Shuffle the data to guarantee integrity of the recorded accuracies
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices,:,:,:]
    y_train = y_train[indices]
    #denoise 1 all, after %4 epochs divide by 10 start with lr 0.01
    #print("Using denoising cost %s " % (dc))

    delete_checkpoints()
    try:
      fa = train(X_train,y_train,X_test,y_test,num_epochs=NUM_EPOCHS,noise_std=params['noise_std'],lr=params['lr'],filter_size=[150,50,30],fc=[30],kernel_size=7,
          denoising_cost=params['denoising_cost'],num_labeled=NUM_LABELED,batch_size=int(params['batch_size']))
      accuracies += [fa]
    except:
      print("Unexpected error:", sys.exc_info()[0])
      accuracies += [0.0]
      status = STATUS_FAIL

  mean, lo, hi = mean_confidence_interval(accuracies)

  run_time = timer() - start

  loss = 1 - (mean/100.0)

  # Write to the csv file ('a' means append)
  of_connection = open(out_file, 'a')
  writer = csv.writer(of_connection)
  writer.writerow([convert_params(params), ITERATION, '{:.1f}'.format(run_time), '{:.5f}'.format(lo), '{:.5f}'.format(mean), '{:.5f}'.format(hi)])

  return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': status}


space = {
      'noise_std': hp.normal('noise_std',0.5, 0.3),
      'lr': hp.loguniform('lr', np.log(0.005), np.log(0.2)),
      'denoising_cost': [hp.uniform('dc1', 0.0, 1.0),hp.uniform('dc2', 0.0, 1.0),hp.uniform('dc3', 0.0, 1.0),
            hp.uniform('dc4', 0.0, 1.0),hp.uniform('dc5', 0.0, 1.0),hp.uniform('dc6', 0.0, 1.0)],
      'batch_size': hp.qnormal('batch_size', 100,20,1)
}


# optimization algorithm
tpe_algorithm = tpe.suggest

# Keep track of results
bayes_trials = Trials()

# File to save first results
out_file = 'results/trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time', '(lo', 'mean', 'high)'])
of_connection.close()

# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
try:
  best = fmin(fn = objective, space = space, algo = tpe.suggest, 
              max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

except:
  print("Unexpected error:", sys.exc_info()[0])

else:
  bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
  print(bayes_trials_results[:1])