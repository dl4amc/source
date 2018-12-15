
# coding: utf-8

# ## Modulation Recognition Example: RML2016.10a Dataset + VT-CNN2 Mod-Rec Network
# 
# More information on this classification method can be found at
# https://arxiv.org/abs/1602.04105
# 
# More information on the RML2016.10a dataset can be found at
# http://pubs.gnuradio.org/index.php/grcon/article/view/11
# 
# Please cite derivative works
# 
# ```
# @article{convnetmodrec,
#   title={Convolutional Radio Modulation Recognition Networks},
#   author={O'Shea, Timothy J and Corgan, Johnathan and Clancy, T. Charles},
#   journal={arXiv preprint arXiv:1602.04105},
#   year={2016}
# }
# @article{rml_datasets,
#   title={Radio Machine Learning Dataset Generation with GNU Radio},
#   author={O'Shea, Timothy J and West, Nathan},
#   journal={Proceedings of the 6th GNU Radio Conference},
#   year={2016}
# }
# ```
# 
# To run this example, you will need to download or generate the RML2016.10a dataset (https://radioml.com/datasets/)
# You will also need Keras installed with either the Theano or Tensor Flow backend working.
# 
# Have fun!

# In[1]:

# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
#get_ipython().magic(u'matplotlib inline')
import os,random
#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)   #disabled because we do not have a hardware GPU
import numpy as np
#import theano as th
#import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
from keras.optimizers import adagrad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.utils import multi_gpu_model
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import tensorflow as tf

# # Dataset setup

# In[2]:

# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("/home/ju10/RML2016.10b_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)


# In[3]:

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples // 2
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#SNR Selection

#functions needed for snr selection
train_SNRs = map(lambda x: lbl[x][1], train_idx)
train_snr = lambda snr: X_train[np.where(np.array(train_SNRs)==snr)]
test_snr = lambda snr: Y_train[np.where(np.array(train_SNRs)==snr)]

#snr pairs selection
#change the snr values correspondingly
X_train_i = train_snr(-20)
Y_train_i = test_snr(-20)
X_train = np.append(X_train_i, train_snr(0),axis=0)
Y_train = np.append(Y_train_i, test_snr(0),axis=0)

#uniformly random selection across all snrs
#comment snr pairs selection and uncomment the code block below
"""
X_train_i = []
X_train_reduced = np.empty((0,2,128))
Y_train_reduced = np.empty((0,10))
for snr in snrs:
  X_train_i = train_snr(snr)
  n_examples = X_train_i.shape[0]
  per_snr_size = n_examples // 128
  train_idx = np.random.choice(range(0,n_examples), size=per_snr_size, replace=False)
  X_train_reduced = np.append(X_train_reduced,X_train_i[train_idx],axis = 0)
  Y_train_i = test_snr(snr)
  Y_train_reduced = np.append(Y_train_reduced, Y_train_i[train_idx],axis=0)

X_train = X_train_reduced
Y_train = Y_train_reduced

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In[4]:
in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp, snrs
classes = mods


# # Build the NN Model

# In[5]:


# Build VT-CNN2 Neural Net model using Keras primitives -- 
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

dr = 0.6 # dropout rate (%)
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv1", activation="relu", data_format="channels_first", padding="valid", filters=256, kernel_size=(1, 3)))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv2", activation="relu", data_format="channels_first", padding="valid", filters=256, kernel_size=(2, 3)))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv3", activation="relu", data_format="channels_first", padding="valid", filters=80, kernel_size=(1, 3)))
model.add(Dropout(dr))

model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(kernel_initializer="glorot_uniform", name="conv4", activation="relu", data_format="channels_first", padding="valid", filters=80, kernel_size=(1, 3)))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))

model.add(Flatten())
model.add(Reshape((1,11200)))
model.add(keras.layers.LSTM(50))
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), init='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
#opt=adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# In[6]:

# Set up some params 
nb_epoch = 200     # number of epochs to train on
batch_size = 1024  # training batch size


# # Train the Model

# In[7]:

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CLDNN_multigpu_0.6.wts.h5'
model = multi_gpu_model(model, gpus=3)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_split=0.25,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)


# # Evaluate and Plot Model Performance

# In[8]:

# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score


# In[9]:
# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('Train_perf.png', dpi=100)	#save image

# In[10]:
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# In[11]:

# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
#plot_confusion_matrix(confnorm, labels=classes)



# In[12]:

# Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="CLDNN Confusion Matrix (SNR=%d)"%(snr))
    #figname = "./CLDNN_result/real_cldnn/CLDNN-confusion-matrix" + str(snr)+".png"
    #plt.savefig(figname)     
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)
    


# In[13]:

# Save results to a pickle file for plotting later
print acc
fd = open('results_cldnn_d0.5.dat','wb')
cPickle.dump( ("CLDNN", 0.5, acc) , fd )


# In[14]:
# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CLDNN Classification Accuracy on RadioML 2016.10 Alpha")
#plt.savefig('./CLDNN_result/real_cldnn/CLDNN-accuracy.png')

