import numpy as np
import tflearn
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import cPickle
import matplotlib.pyplot as plt
import sys
import operator
from numpy import linalg as la 
from math import ceil
maxlen = 128
snrs=""
mods=""
test_idx=""
lbl =""

def gendata(fp, nsamples):
    global snrs, mods, train_idx, test_idx, lbl
    Xd = cPickle.load(open(fp,'rb'))
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    print mods, snrs
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = n_examples//2
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = X[train_idx]
    print('SHAPE OF X_TRAIN:', X_train.shape)
    X_test =  X[test_idx]
    print('SHAPE OF X_TEST:', X_test.shape)
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
    return (X_train,X_test,Y_train,Y_test)


def norm_pad_zeros(X_train,nsamples):
    print "Pad:", X_train.shape
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/la.norm(X_train[i,:,0],2)
    return X_train


def to_amp_phase(X_train,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi
    
    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))
   
    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1) 
    X_train = np.transpose(np.array(X_train),(0,2,1))
    
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    
    
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train, X_test)


xtrain1,xtest1,ytrain1,ytest1 = gendata("./RML2016.10b_dict.dat",maxlen)
print('using version 10b dataset')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Conversion to amp-phase form ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('length of X before to_amp_phase:', xtrain1.shape)
xtrain1,xtest1 = to_amp_phase(xtrain1,xtest1,maxlen)
print('length of X after to_amp_phase:', xtrain1.shape)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xtrain1 = xtrain1[:,:maxlen,:]
xtest1 = xtest1[:,:maxlen,:]

xtrain1 = norm_pad_zeros(xtrain1,maxlen)
xtest1 = norm_pad_zeros(xtest1,maxlen)


X_train = xtrain1
X_test = xtest1

Y_train = np.reshape(ytrain1,(-1,10))
Y_test = np.reshape(ytest1,(-1,10))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
print("--"*50)
print("Training data:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    print(cm)

    print()
    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = mods 
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    return plt

class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.accuracy = 0.0
        self.accs = [0]
        self.patience = 20
        self.patience_epoch_count = 0

    def on_epoch_end(self, training_state):
        print "accuracy2:", training_state.val_acc 
        if self.accuracy<training_state.val_acc:
            self.accuracy = training_state.val_acc 
            print "Model saved:", self.accuracy 
            self.model.save('lstm_large.tfl')
        if training_state.val_acc is not None and training_state.val_acc < max(self.accs):
            self.patience_epoch_count = self.patience_epoch_count + 1
            print("val_acc dropped")
        else:
            self.patience_epoch_count = 0
        if self.patience_epoch_count > self.patience:
            self.patience_epoch_count = 0
            raise StopIteration
        self.accs.append(training_state.val_acc)


#tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.5)
#with tf.device('/gpu:0'):
network = tflearn.input_data(shape=[None, maxlen, 2],name="inp")
network = tflearn.lstm(network, 128, return_seq=True, dynamic=True, dropout=(1, 0.6))
network = tf.transpose(tf.stack(network),[1,0,2])
network = tflearn.lstm(network, 128, return_seq=True, dynamic=True, dropout=(0.6, 0.6))
network = tf.transpose(tf.stack(network),[1,0,2])
network = tflearn.lstm(network, 128, dynamic=True, dropout=(0.6, 1))
network = tflearn.fully_connected(network, len(mods), activation='softmax',name="out")
network = tflearn.regression(network, optimizer='adam',
                 loss='categorical_crossentropy',
                 learning_rate=0.0018)
model = tflearn.DNN(network,tensorboard_verbose=0)

monitorCallback = MonitorCallback(model)


Train = True
if Train:
    try:
        model.fit(X_train, Y_train, n_epoch=500, shuffle=True,show_metric=True,batch_size=500,validation_set=0.25, run_id='radio_lstm', callbacks=monitorCallback)
    except StopIteration:
        print("Caught Exception: Training Stopped")
else:
    model.load('lstm_small.tfl')

classes = mods

acc={}
for snr in snrs:
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    width = 4.1 
    
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("./images/9confmat_"+str(snr)+".pdf")
     
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1 
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor 
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
print(acc)
