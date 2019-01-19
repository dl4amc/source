
# coding: utf-8

# ## Modulation Recognition Example: RML2016.10a Dataset + VT-CNN2 Mod-Rec Network
# More information on this classification method can be found at
# https://arxiv.org/abs/1602.04105
# More information on the RML2016.10a dataset can be found at
# http://pubs.gnuradio.org/index.php/grcon/article/view/11
# Please cite derivative works
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
# To run this example, you will need to download or generate the RML2016.10a dataset (https://radioml.com/datasets/)
# You will also need Keras installed with either the Theano or Tensor Flow backend working.
# Have fun!

"""
Run code for DenseNet Sub-Sampling, PCA and SNR Training experiments by uncommenting the appropriate code blocks
For PCA experiments: Uncomment 'PCA Setup' and 'PCA' code blocks
For Sub-Sampling experiments: Uncomment 'Sub-Sampling Setup' and 1 of the 3 subsampling code blocks
For Individual SNR Training experiments: Uncomment 'SNR Setup' and 'SNR Training' code blocks
For no dimensionality reduction experiments: Run the code as is without uncommenting any code block
"""

# In[1]:
# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
#get_ipython().magic(u'matplotlib inline')
import os,random
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(2)   #disabled because we do not have a hardware GPU
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.regularizers import *
from keras.optimizers import adam
from keras.optimizers import adagrad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sub-Sampling Setup
"""sub_samples = 16    # Number of samples after Sub-Sampling"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PCA Setup
"""from sklearn.decomposition import PCA
pca_rate=4   # Number of samples after PCA
pca = PCA(n_components=pca_rate*2)"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SNR Setup
"""snr_val = -20   # SNR Value to train using"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In[2]:
# Dataset setup
# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("/home/yang1467/DataSet/RML2016.10b_dict.dat",'rb'))
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

# In[4]:
in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Heuristic Sub Sampling
"""n_samples = sub_samples
new_X_train = list()
for wave_idx, wave in enumerate(X_train):
	amp_list = [(iq_idx, ((iq_val[0] ** 2) + (iq_val[1] ** 2) ** 0.5)) for iq_idx, iq_val in enumerate(wave.transpose(1, 0))]
	amp_list.sort(key=lambda x: x[1], reverse=True)
	amp_list = amp_list[:n_samples]
	amp_list.sort(key=lambda x: x[0])
	amp_list = [amp_val[0] for amp_val in amp_list]
	wave = wave.transpose(1, 0)
	wave = wave[amp_list]
	wave = wave.transpose(1, 0)
	new_X_train.append(wave)
X_train = np.stack(new_X_train)

new_X_test = list()
for wave_idx, wave in enumerate(X_test):
	amp_list = [(iq_idx, ((iq_val[0] ** 2) + (iq_val[1] ** 2) ** 0.5)) for iq_idx, iq_val in enumerate(wave.transpose(1, 0))]
	amp_list.sort(key=lambda x: x[1], reverse=True)
	amp_list = amp_list[:n_samples]
	amp_list.sort(key=lambda x: x[0])
	amp_list = [amp_val[0] for amp_val in amp_list]
	wave = wave.transpose(1, 0)
	wave = wave[amp_list]
	wave = wave.transpose(1, 0)
	new_X_test.append(wave)
X_test = np.stack(new_X_test)

print('Number of amplitudes after heuristic sub sampling:', X_train.shape)"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random Sub Sampling
"""n_samples = sub_samples
sample_idx = np.random.choice(range(0,128), size=n_samples, replace=False)
X_train = X_train.transpose((2, 1, 0))
X_train = X_train[sample_idx]
X_train = X_train.transpose((2, 1, 0))
X_test = X_test.transpose((2, 1, 0))
X_test = X_test[sample_idx]
X_test = X_test.transpose((2, 1, 0))"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Uniform Sub Sampling
"""n_samples = sub_samples
sample_idx = [num for num in range(0, 128, 128//n_samples)]
X_train = X_train.transpose((2, 1, 0))
X_train = X_train[sample_idx]
X_train = X_train.transpose((2, 1, 0))
X_test = X_test.transpose((2, 1, 0))
X_test = X_test[sample_idx]
X_test = X_test.transpose((2, 1, 0))"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PCA
"""X_train = X_train.transpose((1, 0, 2))
X_train = np.append(X_train[0], X_train[1], axis=1)
pca_apply = pca.fit(X_train)
print('SHAPE OF X_TRAIN BEFORE PCA', np.shape(X_train))
X_train = pca_apply.transform(X_train)
print('SHAPE OF X_TRAIN AFTER PCA', np.shape(X_train))
X_test = X_test.transpose((1, 0, 2))
X_test = np.append(X_test[0], X_test[1], axis=1)
X_test = pca_apply.transform(X_test)
X_train = np.stack((X_train[:,:len(X_train[0])/2], X_train[:,len(X_train[0])/2:]), axis=1)
X_test = np.stack((X_test[:,:len(X_test[0])/2], X_test[:,len(X_test[0])/2:]), axis=1)
print('FINAL SHAPE OF X_TRAIN', np.shape(X_train))
print('FINAL SHAPE OF X_TEST', np.shape(X_test))
in_shp[1]=pca_rate"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SNR Training
"""X_train = []
Y_train = []
X_train_SNR_idx = []
X_train_SNR = map(lambda x: lbl[x][1], train_idx)
for train_snr, train_index in zip(X_train_SNR, train_idx):
    if train_snr == snr_val:
        X_train_SNR_idx.append(train_index)
X_train = X[X_train_SNR_idx]
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), X_train_SNR_idx))"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In[5]:
# Build the NN Model
dr = 0.5 # dropout rate (%)
inputs=Input(shape=in_shp)
x=Reshape([1]+in_shp)(inputs)
#conv 1
#x=ZeroPadding2D((0, 1))(x)
x=Conv2D(256, 1, 3, border_mode="same", activation="relu", name="conv1", init='glorot_uniform')(x)
x=Dropout(dr)(x)
list_feat=[x]
#conv2
#x=ZeroPadding2D((0, 1))(x)
x=Conv2D(256, 2, 3, border_mode="same", activation="relu", name="conv2", init='glorot_uniform')(x)
x=Dropout(dr)(x)
list_feat.append(x)
m1 = merge(list_feat, mode='concat', concat_axis=1)
#conv 3
#x=ZeroPadding2D((0, 1))(x)
x=Conv2D(80, 1, 3, border_mode="same", activation="relu", name="conv3", init='glorot_uniform')(m1)
x=Dropout(dr)(x)
list_feat.append(x)
m2=merge(list_feat, mode='concat', concat_axis=1)
#conv 4
x=Conv2D(80, 1, 3, border_mode="same",activation="relu", name="conv4", init='glorot_uniform')(m2)
x=Dropout(dr)(x)
list_feat.append(x)
m3=merge(list_feat, mode='concat', concat_axis=1)
#flatten()
x=Flatten()(m3)
#dense 1
x=Dense(256, activation='relu', init='he_normal', name="dense1")(x)
x=Dropout(dr)(x)
#dense 2
x=Dense( len(classes), init='he_normal', name="dense2" )(x)
model = Model(input=[inputs], output=[x])

'''
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, 2, 3, border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
model.add(Dropout(dr))

model.add(Flatten())
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), init='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
'''
#opt=adagrad
#opt=adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy', optimizer='adagrad',metrics=['accuracy'])
model.summary()

# In[6]:
# Set up some params 
nb_epoch = 500     # number of epochs to train on
batch_size = 512  # training batch size

# In[7]:
# Train the Model
# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
#    show_accuracy=False,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)

# In[8]:
# Evaluate and Plot Model Performance
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
plot_confusion_matrix(confnorm, labels=classes)

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
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    figname = "DML-confusion-matrix" + str(snr)+".png"
    plt.savefig(figname)
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)

# In[13]:
# Save results to a pickle file for plotting later
print acc
fd = open('results_cnn2_d0.5.dat','wb')
cPickle.dump( ("CNN2", 0.5, acc) , fd )

# In[14]:
# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.savefig('DML-accuracy.png')
