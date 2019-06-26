"""LadderNet-FC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tYHxfPMmW4a7ogrsiX27G0XdKFVOGY9T
"""

class DataSet:

  def __init__(self,X,y,X_validate,y_validate,X_test,y_test,batch_size=100,architecture=None,denoising_cost=None,
               validate=True,name=None, num_labeled=-1, X_unlabeled = None):

    import os; os.environ['KERAS_BACKEND'] = 'theano'
    from keras.utils import to_categorical
    #Convert to one hot encoding. Same goes for y_test
    self._num_labeled = num_labeled
    self._validate = validate
    if X_unlabeled is not None:
      self._X_unlabeled = X_unlabeled #This is for the case we have truly unlabeled data
      self._X_labeled = X
      self._y_labeled = to_categorical(y)
    else: self._X_unlabeled = None

    if name is not None:
      self._name = name
    else: self._name = 'Unknown'

    self._orig_y = y
    self._y = to_categorical(y)
    self._y_test = to_categorical(y_test)
    self._num_examples = X.shape[0]
    self._X = X

    if not validate:
      self._X_validate = X_test
      self._y_validate = to_categorical(y_test)
    else:
      self._X_validate = X_validate
      self._y_validate = to_categorical(y_validate)

    self._X_test = X_test

    self._current_epoch = 0 #Keep track of the epoch
    self._batch_size = batch_size
    self._curr_index = 0 #This index keeps track of position in the training data
    self._idx = np.arange(0,len(y))

    if architecture is None:
      self._architecture = [X.shape[1], 500, 250, self._y.shape[1]]
    else:
      self._architecture = architecture

    if denoising_cost is None:
      self._denoising_cost = [1000.0, 0.10, 0.10, 0.10]
    else:
      self._denoising_cost = denoising_cost


    if num_labeled != -1 and X_unlabeled is None:
      #Generate a dataset that is labeled and fixed and the rest is an unlabeled dataset
      #For simplicity, when num_labeled = 100, we set aside 100 lab. examples for validation
      #and testing respectively. So it truly corresponds to 100 labeled training examples, but
      #we still need data to verify the model.
      n_classes = len(np.unique(self._orig_y))
      n_from_each_class = int(num_labeled/n_classes)
      indices = np.arange(len(self._y))
      i_labeled = []
      for c in range(n_classes):
            i = indices[self._orig_y==c][:n_from_each_class]
            i_labeled += list(i)


      self._X_labeled = self._X[i_labeled,:]
      self._y_labeled = self._y[i_labeled,:]

      #Sanity check balance
      '''import matplotlib.pyplot as plt
      plt.hist(self._orig_y[i_labeled])
      plt.show()'''

      #Take everything as unlabeled data
      self._X_unlabeled = self._X

  def next_batch(self):
    if self._num_labeled == -1:
      if self._curr_index + self._batch_size < self._X.shape[0]: #shape (numEx,Dim)
        idx = self._idx[self._curr_index:self._curr_index+self._batch_size]
        X_b = self._X[idx]
        y_b = self._y[idx]
        self._curr_index = self._curr_index + self._batch_size
      else:
        #Shuffle data and set current index to batch size
        self._current_epoch = self._current_epoch + 1
        np.random.shuffle(self._idx)
        idx = self._idx[0:self._batch_size]
        self._curr_index = self._batch_size
        X_b = self._X[idx]
        y_b = self._y[idx]
      return (X_b,y_b)

    else:
      #We want to return a stack of labeled and unlab. datapoints. In that case, the labeled images
      #must be drawn evenly distributed from each class
      if self._batch_size > self._num_labeled: #Take all the labeled data points
        idx = np.arange(self._num_labeled)
        np.random.shuffle(idx)
        X_l = self._X_labeled[idx,:]
        y_l = self._y_labeled[idx]
      else:
        idx = np.arange(self._num_labeled)
        np.random.shuffle(idx)
        idx = idx[:self._batch_size]
        X_l = self._X_labeled[idx,:]
        y_l = self._y_labeled[idx]

      idx_ul = np.arange(self._X.shape[0])
      np.random.shuffle(idx_ul)
      idx_ul = idx_ul[:self._batch_size]
      X_ul = self._X_unlabeled[idx_ul,:]

      X_b = np.vstack([X_l, X_ul])
      return (X_b,y_l)

  @property
  def train(self):
    return (self._X, self._y)

  @property
  def validate(self):
    return (self._X_validate, self._y_validate)

  @property
  def test(self):
    return (self._X_test, self._y_test)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def architecture(self):
    return self._architecture

  @property
  def denoising_cost(self):
    return self._denoising_cost

  @property
  def name(self):
    return self._name
  @property
  def num_labeled(self):
    return self._num_labeled
  @property
  def use_validate(self):
    return self._validate

  def diff(self,first, second):
    second = set(second)
    return [item for item in first if item not in second]



  #Class that stores multiple data sets
class DataSets:
  def __init__(self):
    self.datasets = []
    self.current = 0

  def add(self,ds):
    self.datasets.append(ds)

  def get_next(self):
    if self.current == len(self.datasets):
      print("No more datasets.")
      return False
    else:
      self.current += 1
      return self.datasets[self.current - 1]

  def get_dataset(self,name=None):
    if name is not None:
      for ds in self.datasets:
        if ds.name.lower() == name.lower():
          return ds

    else: return self.datasets[0]

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy
!pip install GoogleDriveDownloader
from google_drive_downloader import GoogleDriveDownloader as gdd

def load_pavia():

  gdd.download_file_from_google_drive(file_id='146WN2eZ6Syf-z1KMVRw9GmZdBu_g1JBj',
                                    dest_path='./datasets/paviau.mat', unzip=False)

  gdd.download_file_from_google_drive(file_id='1L9OoAHnLVmPGbfKx8NhEbugxMzE1PG4j',
                                    dest_path='./datasets/paviau_gt.mat', unzip=False)

  X = sio.loadmat('./datasets/paviau.mat')['paviaU']
  y = sio.loadmat('./datasets/paviau_gt.mat')['paviaU_gt']

  return X, y

#Returns dataset with the given hyperparameters
def preprocess_data(name, numComponents, architecture, denoising_cost,batch_size,num_labeled):

  if name == 'pavia':
    X,y = load_pavia()
    n_each = 947
    n_classes = 9

  #Reshape
  X = np.reshape(X, [-1,X.shape[2]])
  y = np.reshape(y, [-1])

  idx = np.arange(len(y))
  idx = idx[(y != 0)]

  X = X[idx,:]
  y = y[idx]-1

  #Shuffle the data
  idx = np.arange(len(y))
  np.random.shuffle(idx)
  X = X[idx,:]
  y = y[idx]

  #Scale the data
  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)

  if numComponents != -1:
    pca = PCA(n_components=numComponents, whiten=True)
    X = pca.fit_transform(X)

  #Downsample
  indices = np.arange(len(y))
  i_labeled = []
  for j in range(n_classes):
    i = indices[y == j][:n_each]
    i_labeled += list(i)

  y = y[i_labeled]
  X = X[i_labeled,:]

  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

  return DataSet(X_train,y_train,None,None,X_test,y_test,batch_size=batch_size,
                    architecture=architecture,denoising_cost=denoising_cost,validate=False,name=name,num_labeled = num_labeled)

import numpy as np
import tensorflow as tf
import math
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.utils import to_categorical
#from tqdm import tqdm

#Split the data in validation and training data
from sklearn.model_selection import train_test_split

class LadderNetwork:
  def __init__(self,id):
    self._id = id


  def delete_checkpoints(self):
    c = str(self._id)
    import shutil
    import os
    if os.path.exists(c + '_checkpoints/') and os.path.isdir(c + '_checkpoints/'):
        shutil.rmtree(c + '_checkpoints/')

  def delete_models(self):
    c = str(self._id)
    import shutil
    import os
    if os.path.exists(c + '_models/') and os.path.isdir(c + '_models/'):
        shutil.rmtree(c + '_models/')


  def get_latest_meta(self):
    c = str(self._id)
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(c + '_checkpoints/') if isfile(join(c + '_checkpoints/', f))]
    for name in onlyfiles:
      if name.lower().endswith('.meta'):
        if int(list(filter(str.isdigit, name))[0]) == int(list(filter(str.isdigit, tf.train.latest_checkpoint(c + '_checkpoints/')))[1]):
          return c + '_checkpoints/' + name


  def get_misclassified(self,X,y,y_cl=None):
    if y_cl is None:
      y_hat = self.predict(X)
      idx = np.arange(len(y_hat))[(y_hat != y)]
      return (X[idx,:], y[idx], y_hat[idx], idx)
    else:
      idx = np.arange(len(y_cl))[(y_cl != y)]
      return (X[idx,:], y[idx], y_hat[idx], idx)

  def fit(self,X_train,y_train,X_validate,y_validate,X_test,y_test,architecture,denoising_cost,num_epochs=150,batch_size=100,
            num_labeled=-1,noise_std=0.3,lr=0.02,decay_after=15,validate=True):
    if validate:
      data = DataSet(X_train,y_train,X_validate,y_validate,X_test,y_test,batch_size,architecture,denoising_cost)
    else:
      data = DataSet(X_train,y_train,None,None,X_test,y_test,batch_size,architecture,denoising_cost,validate=False)
    self.train(data,num_epochs,num_labeled,noise_std,lr,decay_after)


  #Returns accuracy of the trained model
  def train(self,data,num_epochs=150,num_labeled=-1,noise_std=0.3,lr=0.02,decay_after=15):
        tf.reset_default_graph()
        batch_size = data.batch_size

        architecture = data.architecture
        denoising_cost = data.denoising_cost

        #Create placeholders. They will be assigned when we start the session with feed_dict{...}
        inputs = tf.placeholder(tf.float32, shape=(None, architecture[0]), name='inputs')
        outputs = tf.placeholder(tf.float32, name='outputs')

        #This is used to manage different checkpoint/model repositories
        c = str(self._id)

        #Number of layers
        L = len(architecture)-1
        n_examples = data.num_examples

        num_iter = (n_examples//batch_size) * num_epochs

        '''For each layer, we need to keep track of: W the weights for the encoder, V the weights for the decoder
        beta a vector for the encoder, gamma a vector for the encoder, for each neuron in each layer of the decoder
        a 10 size vector A, for the denoising function (the lateral connection)'''

        def vec_init(inits, size, name): #This is for beta and gamma vector in the encoding phase
          return tf.Variable(inits * tf.ones([size]), name=name)

        def mat_init(shape, name): #Use this to initialize the weight matrices, initially random (normal)
          return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

        #The following produces e.g. (784, 1000),(1000, 500),...,(250, 10)
        shapes = list(zip(list(architecture)[:-1], list(architecture[1:])))

        #Initialize a dictionary of the weights. This makes it easy to keep track of the different dimensions and keep
        #Them in one data structure
        weights = {'W': [mat_init(s, "W") for s in shapes],
                 'V': [mat_init(s[::-1], "V") for s in shapes],
                 # batch normalization parameter to shift the normalized value
                 'beta': [vec_init(0.0, architecture[l+1], "beta") for l in range(L)],
                 # batch normalization parameter to scale the normalized value
                 'gamma': [vec_init(1.0, architecture[l+1], "beta") for l in range(L)]}


        #Some helper functions
        join = lambda l, u: tf.concat([l, u], 0)
        labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
        unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
        split_lu = lambda x: (labeled(x), unlabeled(x))

        #Placeholder for a boolean variable if we are training or not
        training = tf.placeholder(tf.bool,name='training')

        #This keeps a moving average of the layers std and mean. This is valuable if the input is clean.
        #If the input is corrupted, we have to use the nn.moment variant to approximate the std and mean.
        ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        bn_assigns = []  # this list stores the updates to be made to average mean and variance

        #Normalize the batch. This is used for non-clear inputs.
        def batch_normalization(batch, mean=None, var=None):
          if mean is None or var is None:
              mean, var = tf.nn.moments(batch, axes=[0])
          return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

        #Average mean and variance of all layers for the labeled data. When we only have 100 labeled data in total
        #The batch var and mean is not enough. This is why we have to update the mean and std for the labeled data.
        #Use this var and mean for batch normalization later
        running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in architecture[1:]]
        running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in architecture[1:]]

        #Batch normalize + update average mean and variance of layer l
        def update_batch_normalization(batch, l):
          mean, var = tf.nn.moments(batch, axes=[0])
          assign_mean = running_mean[l-1].assign(mean)
          assign_var = running_var[l-1].assign(var)
          bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
          with tf.control_dependencies([assign_mean, assign_var]):
              return (batch - mean) / tf.sqrt(var + 1e-10)


        '''Now define the encoder.
        The encoder serves as a noise introducing and clean encoder. At each leavel, we do the following:

        For the corrupted case:
        Assign the z0 <- x(n)+noise then h0 <- z0 (no batch normalization)
        For all the layers:
        zl = batchnorm(W*hl-1)+noise /// hl <- actvation(gamma had.prod. (zl+beta))
        At the end: Output y <-hL, we use the corruped output for the cost (this serves as regularization)

        For the clean encoder:
        z0 <- x(n) then h0 <- z0
        For all layers:
        z_pre_l <- Wl*h(l-1) //// mean(l) <- batchmean(z_pre_l) /// std(l) <- batchstd(z_pre_l) //// z_l <- batchnorm(z_pre)
        h_l <- activation(gamma had.prod. (z_l + beta))
        The Mean and std are used for batch normalisation in the decoder
        h is used as the input to the next layer and the final classification
        z is used for the cost function. We want to min. ||z_clean - z_noisy_recond|| for all layers
        '''
        def encoder(inputs,noise_std):
          h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std #Clean input if the noise std is set to zero
          d = {} #Store normalized preactivation z_l, h_l, mean, std

          #Initialize the dictionary that stores the data. Note that the data is stored seperately
          #The speration is because we still want to know for which examples we have the labels
          d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
          d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

          #Initialize the lowest layer with h. We do not have a transformation there.
          d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)

          #Loop through all the layers. Doing forward propagation and updating the values we need to keep track of.
          for l in range(1, L+1): #Max. index: L

              #Split the data that was joined before (TODO: Check if this can be done at end of loop)
              d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
              #Calculate the preactivation
              z_pre = tf.matmul(h, weights['W'][l-1])
              #Split into labeled and unlabeled examples
              z_pre_l, z_pre_u = split_lu(z_pre)
              #Caculate the mean and variance of the unlabeled examples, this is needed in the decoder phase when normalizing the
              m, v = tf.nn.moments(z_pre_u, axes=[0])

              def training_batch_norm():
                  # Training batch normalization
                  # batch normalization for labeled and unlabeled examples is performed separately
                  if noise_std > 0:
                      # Corrupted encoder, do not update the mean and std of the layer
                      # batch normalization + noise
                      z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v)) #CHANGE
                      z += tf.random_normal(tf.shape(z_pre)) * noise_std
                  else:
                      # Clean encoder
                      # batch normalization + update the average mean and variance using batch mean and variance of
                      # labeled examples
                      z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
                  return z

              def eval_batch_norm():
                  # Evaluation batch normalization
                  # obtain average mean and variance and use it to normalize the batch
                  mean = ewma.average(running_mean[l-1])
                  var = ewma.average(running_var[l-1])
                  z = batch_normalization(z_pre, mean, var)
                  return z
              #If we are traning, use the training batch norm (we also have labeled data)
              z = tf.cond(training, training_batch_norm, eval_batch_norm)

              if l == L:
                #Convert z and apply softmax for the last layer. (TODO: Only for prediction or if we pass through encoder?)
                h = tf.nn.softmax(weights['gamma'][l-1] * (z+weights['beta'][l-1]))
              elif l == L-1:
                h = tf.nn.relu(z + weights['beta'][l-1])
              else:
                h = tf.nn.relu(z + weights['beta'][l-1]) #TODO: No gamma?

              #We split z and save the mean and variance of the unlabeled data for the decoder, where it is needed
              d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
              d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v
          #Return the values at each layer. h is the output used (y) either corrupted or clean.
          d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
          return h, d

        def get_activation(inputs,layer): #E.g. for last layer (not the 10 dim one) put L-1
          h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std #Clean input if the noise std is set to zero
          d = {} #Store normalized preactivation z_l, h_l, mean, std

          #Initialize the dictionary that stores the data. Note that the data is stored seperately
          #The speration is because we still want to know for which examples we have the labels
          d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
          d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

          #Initialize the lowest layer with h. We do not have a transformation there.
          d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)

          #Loop through all the layers. Doing forward propagation and updating the values we need to keep track of.
          for l in range(1, L+1): #Max. index: L

              #print("Layer %s: ,%s -> %s" % (l,architecture[l-1],architecture[l]))
              #Split the data that was joined before (TODO: Check if this can be done at end of loop)
              d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
              #Calculate the preactivation
              z_pre = tf.matmul(h, weights['W'][l-1])
              #Split into labeled and unlabeled examples
              z_pre_l, z_pre_u = split_lu(z_pre)
              #Caculate the mean and variance of the unlabeled examples, this is needed in the decoder phase when normalizing the
              m, v = tf.nn.moments(z_pre_u, axes=[0])
              m_l, v_l = tf.nn.moments(z_pre_l, axes=[0]) #CHANGE

              #If we are traning, use the training batch norm (we also have labeled data)
              mean = ewma.average(running_mean[l-1])
              var = ewma.average(running_var[l-1])
              z = batch_normalization(z_pre, mean, var)

              if l == L:
                #Convert z and apply softmax for the last layer. (TODO: Only for prediction or if we pass through encoder?)
                h = tf.nn.softmax(weights['gamma'][l-1] * (z+weights['beta'][l-1]))
                return h
              elif l == layer:
                h = tf.nn.relu(z + weights['beta'][l-1])
                return h
              else:
                h = tf.nn.relu(z + weights['beta'][l-1]) #TODO: No gamma?




        last_layer_activation = get_activation(inputs, L-1)
        last_layer_activation = tf.identity(last_layer_activation, name='last_layer_activation')

        #Noise pass
        y_c, corr = encoder(inputs, noise_std)
        #Clean pass, do the clean pass after the noisy pass
        y, clean = encoder(inputs, 0.0)
        y = tf.identity(y, name="y")

        #This is the function that performs the denoising
        def g_gauss(z_c, u, size):
          wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
          a1 = wi(0., 'a1')
          a2 = wi(1., 'a2')
          a3 = wi(0., 'a3')
          a4 = wi(0., 'a4')
          a5 = wi(0., 'a5')

          a6 = wi(0., 'a6')
          a7 = wi(1., 'a7')
          a8 = wi(0., 'a8')
          a9 = wi(0., 'a9')
          a10 = wi(0., 'a10')

          mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
          v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

          z_est = (z_c - mu) * v + mu
          return z_est

        #Decoder
        '''
        The decoder does the following:
        For l=L --> 0:
        If we are at the top (l==L) we do u <- batchnorm(h_L_corrupted)
        Else:
        u <- batchnorm(V*z_recon)
        What is z_recon?:  Take the previously u and the noisy z_l perform denoising function
        z_recon <- g(z_l_noisy, u_l) /// u_l was previously assigned
        z_recond_BN <- batch_normalize(z_recon_l, m, v) where m and v are the mean and variance from the layer z
        from the clean run.
        It is important to first make the corr. run and then the clean run. Otherwise we will use the
        m and v from the noisy run.
        '''
        z_est = {} #This corresponds to z_hat in the paper. This is not the batch normalize version
        d_cost = [] #Store the reconstruction cost of each layer

        for l in range(L, -1, -1):
          #Get the clean and noisy layer values from the encoder run. z is used for reconstruction and
          #z_c is used for denoising (lateral connection) and z is used for cost function
          z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
          #Get the mean and variance from the clean run at the current layer l
          #TODO: Why only form the unlabeled data?
          m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
          if l == L: #Initial assignment of u is the h_corr of the previous run through the encoder (noisy)
              u = unlabeled(y_c)
          else:
            #Just multiply with the weights and normalize the batch using the batch std and mean
            u = tf.matmul(z_est[l+1], weights['V'][l])
          u = batch_normalization(u)
          #Apply denoising function (lateral connection)
          z_est[l] = g_gauss(z_c, u, architecture[l]) #TODO: Are these weights learned when we are reinit. the vars?
          z_est_bn = (z_est[l] - m) / v #Calculate the BN but don't save it. We only need it for the cost.
          #Append the cost of this layer to d_cost
          d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / architecture[l]) * denoising_cost[l])


        #Calculate total unsupervised cost by adding the denoising cost of all layers
        u_cost = tf.add_n(d_cost)

        #Use the corrupted output from the encoder as a prediction
        y_N = labeled(y_c)
        #Apply the supervised cost definition of true output * log(output noisy encoder + last layer)
        cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1))
        loss = cost + u_cost  # total cost

        pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1),name='pred_cost')  # cost used for prediction

        #Use y for final classification. Use y_corr if we are training
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1), name='correct_prediction')

        accuracy = tf.multiply(tf.reduce_mean(tf.cast(correct_prediction, "float")),tf.constant(100.0),name='accuracy')

        learning_rate = tf.Variable(lr, trainable=False)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*bn_assigns)
        with tf.control_dependencies([train_step]):
           train_step = tf.group(bn_updates)

        if CHECKPOINT:
          saver = tf.train.Saver()

        sess = tf.Session()

        i_iter = 0
        if CHECKPOINT:
          ckpt = tf.train.get_checkpoint_state(c + '_checkpoints/')
          if ckpt and ckpt.model_checkpoint_path:
            print("Found checkpont! Restore...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            i_iter = (epoch_n+1) * (n_examples/batch_size)
            print("Restored Epoch %s" % epoch_n)
          else:
            if not os.path.exists(c + '_checkpoints'):
              os.makedirs(c + '_checkpoints')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
          init = tf.global_variables_initializer()
          sess.run(init)

        epoch_n = 0
        old_acc = 0.0
        for i in range(int(i_iter),num_iter):
          #Get the next batch of batch size (images and labels)
          images, labels = data.next_batch()
          sess.run(train_step, feed_dict={inputs: images, outputs: labels, training: True})
          if (i > 1) and ((i+1) % (num_iter//num_epochs) == 0):
            epoch_n = i // (n_examples//batch_size)
            acc = sess.run(accuracy, feed_dict={inputs: data.validate[0], outputs: data.validate[1], training: False})
            #print("Epoch: %s, Accuracy: %s" % (epoch_n, acc))

            if (epoch_n+1) >= decay_after:
                # decay learning rate
                # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) // (num_epochs - decay_after))
                ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
                ratio = max(0, ratio // (num_epochs - decay_after))
                sess.run(learning_rate.assign(lr * ratio))
            if acc > old_acc:
              #print("Improved accuracy. Save...")
              if CHECKPOINT:
                saver.save(sess, c + '_checkpoints/model.ckpt', epoch_n)
                model_inputs = {
                    "inputs_placeholder":inputs,
                    "outputs_placeholder":outputs
                }
                model_outputs = {
                    "accuracy": accuracy,
                    "clean_output": y,
                    "last_layer_activation": last_layer_activation
                }
                self.delete_models()
                tf.saved_model.simple_save(sess, c + '_models/',model_inputs,model_outputs)
                old_acc = acc

            #print("Created checkpoint.")
        fa = sess.run(accuracy, feed_dict={inputs: data.test[0], outputs: data.test[1], training: False})
        print("Final accuracy is: %s" % fa)
        writer = tf.summary.FileWriter('./log/pshnn_ladder', sess.graph)

        model_inputs = {
            "inputs_placeholder":inputs,
            "outputs_placeholder":outputs
        }
        model_outputs = {
            "accuracy": accuracy,
            "clean_output": y,
            "last_layer_activation": last_layer_activation
        }
        self.delete_models()
        tf.saved_model.simple_save(sess, c + '_models/',model_inputs,model_outputs)
        return fa


  def predict(self,X,y=None):
    c = str(self._id)
    from tensorflow.python.saved_model import tag_constants
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(self.get_latest_meta())
    restored_graph = tf.get_default_graph()
    with restored_graph.as_default():
      with tf.Session() as sess:
        tf.saved_model.loader.load(
          sess,
          [tag_constants.SERVING],
          c + '_models/'
        )
        inputs_placeholder = restored_graph.get_tensor_by_name('inputs:0')
        outputs_placeholder = restored_graph.get_tensor_by_name('outputs:0')
        training = restored_graph.get_tensor_by_name('training:0')
        clean_output = restored_graph.get_tensor_by_name('y:0')

        out = sess.run(clean_output, feed_dict={inputs_placeholder: X, training:False})
        res = [np.argmax(out[i]) for i in range(out.shape[0])]
        if y is not None:
          accuracy = restored_graph.get_tensor_by_name('accuracy:0')
          acc = sess.run(accuracy, feed_dict={inputs_placeholder: X, outputs_placeholder: to_categorical(y), training:False})
          return (np.asarray(res),acc)
        else:
          return np.asarray(res)

  def get_last_layer_activation(self,X,dim):
    c = str(self._id)
    act = np.zeros((X.shape[0],dim))
    from tensorflow.python.saved_model import tag_constants
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(self.get_latest_meta())
    restored_graph = tf.get_default_graph()
    with restored_graph.as_default():
      with tf.Session() as sess:
        tf.saved_model.loader.load(
          sess,
          [tag_constants.SERVING],
          c + '_models/'
        )
        inputs_placeholder = restored_graph.get_tensor_by_name('inputs:0')
        outputs_placeholder = restored_graph.get_tensor_by_name('outputs:0')
        training = restored_graph.get_tensor_by_name('training:0')
        last_layer_activation = restored_graph.get_tensor_by_name('last_layer_activation:0')
        out = sess.run(last_layer_activation, feed_dict={inputs_placeholder: X, training:False})
        return out


  def get_activation(self,X):
    c = str(self._id)
    from tensorflow.python.saved_model import tag_constants
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(self.get_latest_meta())
    restored_graph = tf.get_default_graph()
    with restored_graph.as_default():
      with tf.Session() as sess:
        tf.saved_model.loader.load(
          sess,
          [tag_constants.SERVING],
          c + '_models/'
        )
        inputs_placeholder = restored_graph.get_tensor_by_name('inputs:0')
        outputs_placeholder = restored_graph.get_tensor_by_name('outputs:0')
        training = restored_graph.get_tensor_by_name('training:0')
        clean_output = restored_graph.get_tensor_by_name('y:0')
        out = sess.run(clean_output, feed_dict={inputs_placeholder: X, training:False})
        return out



import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#Pavia pipeline
# nl = [45,90,270,450,-1]
# bs = [45,90,100,100,100]
# nc = [-1,-1,-1,-1,30]
nl = [15]
bs = [135]
nc = [-1]

epochs = 20
noise_std = 0.1
learning_rate = 0.005
decay_after = 30

CHECKPOINT = True

for j in range(5):
  num_labeled = nl[j]
  batch_size = bs[j]
  n_components = nc[j]

  accuracies = []
  for _ in range(20):

    if n_components == -1:
      fd = 103
    else: fd = n_components
    ds = preprocess_data('pavia',n_components,[fd,300,200,100,100,9],[10.0,1.0,0.1,0.1,0.1,0.1],batch_size,num_labeled)

    #If running for the first time, comment out the following two lines.
    #nn.delete_checkpoints()
    #nn.delete_models()
    tf.logging.set_verbosity(tf.logging.WARN)
    nn = LadderNetwork(1)

    acc = nn.train(ds,epochs,num_labeled,noise_std,learning_rate,decay_after)
    accuracies += [acc]
    print(accuracies)

  mean, lo, hi = mean_confidence_interval(accuracies)
  print("Confidence interval is [%s ; %s ; %s]" % (lo,mean,hi))