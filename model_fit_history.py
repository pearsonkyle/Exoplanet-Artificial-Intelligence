import pickle
import numpy as np
from sklearn import preprocessing
import pickle
import pywt
import numpy as np

#from generate_data import load_data #need ELCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers.pooling import AveragePooling1D
from keras.layers.advanced_activations import PReLU as PRELU
from keras.callbacks import History
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

# db2
def wavy(Xt,wavelet='db2'):
    cA,cD = pywt.dwt(Xt[0],wavelet,'symmetric')
    size = cA.shape[0]+cD.shape[0]

    X = np.zeros((Xt.shape[0],size))
    for i in range(Xt.shape[0]):
        cA, cD = pywt.dwt(Xt[i], wavelet)
        X[i] = list(cA) + list(cD)
    return X


arr = lambda x : np.array( list(x),dtype=np.float )
def load_data(fname='transit_data.pkl',categorical=False,whiten=True,DIR='pickle_data/'):

    data = pickle.load(open(DIR+fname,'rb'))

    # convert to numpy array fo float type from object type
    pvals = arr(data['results'][:,0])
    transits = arr(data['results'][:,1])
    null = arr(data['results'][:,2])

    X = np.vstack([transits,null])
    y = np.hstack([np.ones(transits.shape[0]), np.zeros(null.shape[0])] )

    if categorical: y = np_utils.to_categorical(y, np.unique(y).shape[0] )
    if whiten: X = preprocessing.scale(X,axis=1)

    return X,y,pvals,data['keys'],data['time']


def pn_rates(model,X,y):
    y_pred = model.predict_classes(X)

    pos_idx = y==1
    neg_idx = y==0

    tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
    fn = np.sum(y_pred[pos_idx]==0)/y_pred.shape[0]

    tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
    fp = np.sum(y_pred[neg_idx]==1)/y_pred.shape[0]

    return fn,fp


def make_wave(maxlen):
    model = Sequential()
    # conv1
    model.add(Dense(64,input_dim=maxlen, kernel_initializer='he_normal',bias_initializer='zeros' ) )
    model.add(PRELU())
    model.add(Dropout(0.25))

    model.add(Dense(32))
    model.add(PRELU())

    model.add(Dense(8))
    model.add(PRELU())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                optimizer=SGDsolver,
                metrics=['accuracy'])
    return model

def make_nn(maxlen):
    model = Sequential()    # conv1
    model.add(Dense(64,input_dim=maxlen, kernel_initializer='he_normal',bias_initializer='zeros' ) )
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(PRELU())
    model.add(Dropout(0.25))


    model.add(Dense(32, kernel_initializer='he_normal',bias_initializer='zeros'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(PRELU())
    #model.add(Dropout(0.5))


    model.add(Dense(8, kernel_initializer='he_normal',bias_initializer='zeros'))
    #model.add(Activation('relu'))
    model.add(PRELU())


    model.add(Dense(1, kernel_initializer='he_normal',bias_initializer='zeros'))
    model.add(Activation('sigmoid'))


    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                optimizer=SGDsolver,
                metrics=['accuracy'])

    return model

def make_cnn(maxlen):

    pool_length = 3

    model = Sequential()
    model.add(Conv1D(kernel_size=5, filters=8,
                            activation=PRELU(),
                            input_shape=(maxlen,1),
                            #strides=1, # same as subsample_length?
                            name='conv1',
                            padding="valid"))
    model.add(AveragePooling1D(pool_size=pool_length))


    model.add(Conv1D(kernel_size=5, filters=8,
                            activation=PRELU(),
                            input_shape=(maxlen,1),
                            #strides=1, # same as subsample_length?
                            name='conv2',
                            padding="valid"))
    model.add(AveragePooling1D(pool_length=pool_length))

    # conv2
    #model.add(Convolution1D(nb_filter=4,
    #                        filter_length=filter_length,
    #                        border_mode='valid',
    #                        activation='linear',
    #                        subsample_length=1,
    #                        name='conv2'))
    #model.add(AveragePooling1D(pool_length=pool_length))
    model.add(Flatten())

    model.add(Dense(64, kernel_initializer='he_normal',bias_initializer='zeros'))
    model.add(PRELU())
    #model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32, kernel_initializer='he_normal',bias_initializer='zeros'))
    model.add(PRELU())
    #model.add(Activation('relu'))

    model.add(Dense(8, kernel_initializer='he_normal',bias_initializer='zeros'))
    model.add(PRELU())
    #model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                optimizer=SGDsolver,
                metrics=['accuracy'])
    return model

def make_svm(maxlen):

    model = Sequential()

    model.add(Dense(1,input_dim=maxlen, kernel_initializer='he_normal',bias_initializer='zeros'))
    model.add(Activation('sigmoid'))

    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0, nesterov=True)
    model.compile(loss='binary_crossentropy',
                optimizer=SGDsolver,
                metrics=['accuracy'])
    return model

if __name__ == "__main__":

    # load DATA
    X_train,y_train,pvals,keys,time = load_data('transit_data_train.pkl',whiten=True)
    X_test,y_test,pvals,keys,time = load_data('transit_data_test.pkl',whiten=True)

    Xw_train = wavy(X_train)
    Xw_test = wavy(X_test)

    Xc_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    Xc_test = X_test.reshape((X_test.shape[0],X_train.shape[1],1))

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Training
    batch_size = 128
    nb_epoch = 30

    models = {
        'MLP':make_nn(X_train.shape[1]),
        'Wavelet MLP':make_wave(Xw_train.shape[1]),
        'CNN 1D':make_cnn(Xc_train.shape[1]),
        'SVM':make_svm(X_train.shape[1])
    }

    inputs = {
        'MLP':X_train,
        'Wavelet MLP':Xw_train,
        'CNN 1D':Xc_train,
        'SVM':X_train
    }

    tests = {
        'MLP':X_test,
        'Wavelet MLP':Xw_test,
        'CNN 1D':Xc_test,
        'SVM':X_test
    }

    colors = {
        'MLP':'red',
        'Wavelet MLP':'green',
        'CNN 1D':'blue',
        'SVM': 'orange'
    }
    historys = {}

    f,ax = plt.subplots(2)

    # train each model
    for k in models.keys():
        print(k)
        history = models[k].fit(inputs[k], y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=0, validation_split=0.0, validation_data=None)
        historys[k] = history

        score = models[k].evaluate(inputs[k], y_train, verbose=0)
        fn,fp = pn_rates(models[k],inputs[k],y_train)
        print('\nTrain loss:', score[0])
        print('Train accuracy:', score[1])
        print('Train FP:',fp)
        print('Train FN:',fn)

        score = models[k].evaluate(tests[k], y_test, verbose=0)
        fn,fp = pn_rates(models[k],tests[k],y_test)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])
        print('Test FP:',fp)
        print('Test FN:',fn)

        ax[0].plot(1+np.arange(len( history.history['loss'])), np.log2( history.history['loss'] ),'k-',label='{}'.format(k),color=colors[k])
        ax[1].plot(1+np.arange(len( history.history['loss'])),np.log2(1-np.array(history.history['acc'])),'r-',label='{}'.format(k),color=colors[k])

        models[k].save('models/{}_transit.h5'.format(k))

    ax[0].set_title('Training Metrics')
    ax[0].set_ylabel(r'$Log_{2}$( Loss )' )
    ax[1].set_ylabel(r'$Log_{2}$( 1-Accuracy )' )
    ax[0].set_xlabel('Training Epoch')
    ax[1].set_xlabel('Training Epoch')
    ax[0].legend(loc='best')
    plt.show()
