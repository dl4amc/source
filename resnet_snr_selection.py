from keras import layers
from keras import models
import os, random, keras, cPickle
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import AlphaDropout
from keras.optimizers import adam
from keras.utils import multi_gpu_model
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# data pre-processing
Xd = cPickle.load(open("./RML2016.10b_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
print("length of snr",len(snrs))
print("length of mods",len(mods))
X = [] 
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
print("shape of X", np.shape(X))
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
print("shape of Y_train")
print(Y_train.shape)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In[4]:
print('training started')
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


# Resnet Architecture
# why do they not use batchnorm?
def residual_stack(x):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='linear')(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y
  
  x = layers.Conv1D(32, data_format='channels_first',kernel_size=1, padding='same',activation='linear')(x)
  x = layers.BatchNormalization()(x)
  x = residual_unit(x)
  x = residual_unit(x)
  # maxpool for down sampling
  x = layers.MaxPooling1D(data_format='channels_first')(x)
  return x

inputs=layers.Input(shape=in_shp)
x = residual_stack(inputs)  # output shape (32,64)
x = residual_stack(x)    # out shape (32,32)
x = residual_stack(x)    # out shape (32,16)
x = Flatten()(x)
x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense1")(x)
x = AlphaDropout(0.25)(x)
x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense2")(x)
x = AlphaDropout(0.25)(x)
x = Dense(len(classes),kernel_initializer="he_normal", activation="softmax", name="dense3")(x)
x_out = Reshape([len(classes)])(x)
model = models.Model(inputs=[inputs], output=[x_out])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# Set up some params 
nb_epoch = 100     # number of epochs to train on
batch_size = 1024  # training batch size


# # Train the Model

# In[7]:

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'simulated_resnet_10b.wts.h5'
model = multi_gpu_model(model, gpus=[0,1,2])
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
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print(score)

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
    #plt.figure()
    #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "Overall Accuracy: ", cor / (cor+ncor)
    acc[snr] = 1.0*cor/(cor+ncor)
    


# In[13]:

# Save results to a pickle file for plotting later
print acc
fd = open('results_resnet_10b.dat','wb')
cPickle.dump( acc , fd )


# In[14]:

# Plot accuracy curve
plt.plot(snrs, map(lambda x: acc[x], snrs))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("resnet Classification Accuracy on 2018.01_mod24_1024")

