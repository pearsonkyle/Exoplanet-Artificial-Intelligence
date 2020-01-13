import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from ELCA import lc_fitter, transit


def pn_rates(model,X,y):
    y_pred = np.round( model.predict(X) )

    pos_idx = y==1
    neg_idx = y==0

    tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
    fn = np.sum(y_pred[pos_idx]==0)/y_pred.shape[0]

    tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
    fp = np.sum(y_pred[neg_idx]==1)/y_pred.shape[0]

    return fn,fp

def make_cnn(maxlen):
    
    input_local = Input(shape=(maxlen,1))
    x = Conv1D(16, 5, strides=1)(input_local)
    #x = Conv1D(16, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)
    x = Conv1D(8, 5, strides=1)(x)
    #x = Conv1D(8, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)
    
    xf = Flatten()(x)
    z = Dense(64, activation='relu')(xf)
    #z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    z = Dense(8, activation='relu')(z)

    output = Dense(1, activation='sigmoid', name='main_output')(z)
    model = Model(inputs=input_local, outputs=output)
    
    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                optimizer=SGDsolver,
                metrics=['accuracy'])
    return model

def generate_negatives(residuals, N=10000):
    negative = []

    # generate some negative examples 
    for i in range(10000):

        # choose random sample of time, ensure we have enough data points ``
        length = 0
        while (length != 180):
            di = np.random.choice( np.arange(len(times)) )

            # make sure we get enough data for NN
            ti = np.random.choice( np.arange(len(times[di])) )
            start_t = times[di][ti]
            end_t = times[di][ti] + 2*180/(24*60)
            tmask = (times[di] > start_t) & (times[di] < end_t)
            length = tmask.sum()

        negative.append(residuals[di][tmask])

    return negative 

def generate_positives(residuals, pars, N=10000, minsnr=1.5, snrscale=2):
    positive = []
    snrs = []

    # generate some positive examples 
    for i in range(10000):

        # choose random sample of time, ensure we have enough data points ``
        length = 0
        while (length != 180):
            di = np.random.choice( np.arange(len(times)) )

            # make sure we get enough data for NN
            ti = np.random.choice( np.arange(len(times[di])) )
            start_t = times[di][ti]
            end_t = times[di][ti] + 2*180/(24*60)
            tmask = (times[di] > start_t) & (times[di] < end_t)
            length = tmask.sum()

        tlength = 0 
        # make sure the transit length is sufficient 
        while (tlength < 15):
            std = np.std( residuals[di][tmask] )
            snr = minsnr+np.random.uniform()*snrscale
            snrs.append( snr )

            # create parameters for injected data set 
            pars['ar'] = np.random.uniform()*7 + 7
            pars['per'] = np.random.uniform()*4 + 2
            pars['tm'] = np.median( times[di][tmask] ) + np.random.random()/24/60
            pars['rp'] = ( np.abs(snr)*std)**0.5 

            tmodel = transit(time=times[di][tmask], values=pars)

            if (snr<0):
                tmodel *= -1 
                tmodel += 2
            
            tlength = (tmodel!=1).sum()

        # create data and assess transit probability 
        data = residuals[di][tmask] * tmodel    
        pdata = preprocessing.scale( data )
        positive.append(pdata)

    return snrs, positive 

if __name__ == "__main__":
    
    data_fp = "TESS"

    try:
        # package results from Pearson 2019 

        # loop through TESS data to get residuals 
        FILENAME = 'TESS/TESSdata.pkl'
        tess_data = pickle.load(open(FILENAME,'rb'))

        # alloc data arrays
        times = []
        residuals = []; fluxs = []
        tics = []

        # loop through existing TESS data for precomputed
        for index, row in tess_data.iterrows():
            
            if os.path.exists( row.lcdata ):
                data = pickle.load(open(row.lcdata,'rb'))
                if not 'r' in data: # check to see if residuals have been calculated 
                    continue 
                    
                tics.append(row.tic_id)
                times.append(data['t'])
                residuals.append(data['r'])
                fluxs.append(data['f'])

        pickle.dump({'tic':tics,'time':times,'residual':residuals, 'fluxs':data['f']}, open('TESS_residual.pkl','wb'))

    except: 
        print('loading data from pickle file')
        sv = pickle.load( open('TESS_residual.pkl','rb') )
        residuals = sv['residual']
        times = sv['time']

    # create some random data yo
    pars = {'ar': 15.9187, 'u1': 0.4242, 'u2': 0.1400, 'inc': 89.325, 'ecc': 0, 'ome': 0, 'a0': 1, 'a1': 0, 'a2': 0}

    # create training data 
    #negative = generate_negatives(data, residuals, 10000)
    snrs, positive = generate_positives(residuals, pars, 20000, minsnr=1, snrscale=2)

    # create some low snr transits to train on as negative, shouldn't detect a transit beneath the noise
    snrs2, negative  = generate_positives(residuals, pars, 20000, minsnr=-1, snrscale=1.25)

    X = np.vstack([positive, negative])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = np.hstack([np.ones(len(positive)), np.zeros(len(negative))])

    nb_epoch = 25
    batch_size = 32
    cnn = make_cnn(180)
    history = cnn.fit(X, y, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_split=0.0, validation_data=None)

    cnn.save_weights('tess_cnn1d.h5')

    score = cnn.evaluate(X, y, verbose=1)
    fn,fp = pn_rates(cnn,X,y)
    print('\nTrain loss:', score[0])
    print('Train accuracy:', score[1])
    print('Train FP:',fp)
    print('Train FN:',fn)

    # test the network 
    negative = generate_negatives( residuals, 10000)
    snrs, positive = generate_positives( residuals, pars, 10000, minsnr=0.05, snrscale=2)
    snrs2, negative  = generate_positives( residuals, pars, 10000, minsnr=-1, snrscale=1)
    Xt = np.vstack([positive,negative])
    Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], 1))
    yt = np.hstack([np.ones(len(positive)), np.zeros(len(negative))])

    score = cnn.evaluate(Xt, yt, verbose=1)
    fn,fp = pn_rates(cnn,Xt,yt)
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test FP:',fp)
    print('Test FN:',fn)

    y_pos = cnn.predict(Xt[:10000])
    y_neg = cnn.predict(Xt[10000:])




    plt.plot( snrs, y_pos[:,0],'g.',label='Test Positive',alpha=0.5)
    plt.plot( snrs2, y_neg[:,0],'r.',label='Test Negative',alpha=0.5)
    plt.fill_between([1,2],0,1, facecolor='green',label='Training Positive', alpha=0.25)
    plt.fill_between([-1,0.25],0,1, facecolor='purple',label='Training Negative', alpha=0.25)
    
    plt.xlabel('Transit SNR')
    plt.ylabel('Transit Probability')
    plt.title('CNN Transit Recovery using TESS Noise')

    bsnrs = np.linspace(min(snrs),max(snrs),21)
    bsnr = np.zeros(20)
    bavg = np.zeros(20)
    bstd = np.zeros(20)
    for i in range(bsnr.shape[0]):
        smask = (snrs > bsnrs[i] ) & (snrs<bsnrs[i+1])
        bavg[i] = np.mean( np.array(y_pos)[smask] )
        bsnr[i] = np.mean( np.array(snrs)[smask] )
        bstd[i] = np.std( np.array(y_pos)[smask] )

    plt.errorbar(bsnr,bavg,yerr=bstd,ls='none',marker='.',label='Average',color='black',alpha=0.75)
    #plt.fill_between(bsnr,bavg-bstd,bavg+bstd,alpha=0.75,color='black',label='Average')


    bsnrs = np.linspace(min(snrs2),max(snrs2),13)
    bsnr = np.zeros(12)
    bavg = np.zeros(12)
    bstd = np.zeros(12)
    for i in range(bsnr.shape[0]):
        smask = (snrs2 > bsnrs[i] ) & (snrs2<bsnrs[i+1])
        bavg[i] = np.mean( np.array(y_neg)[smask] )
        bsnr[i] = np.mean( np.array(snrs2)[smask] )
        bstd[i] = np.std( np.array(y_neg)[smask] )

    plt.errorbar(bsnr,bavg,yerr=bstd,ls='none',marker='.',color='black',alpha=0.75)

    #plt.fill_between(bsnr,bavg-bstd,bavg+bstd,alpha=0.75,color='black')
    plt.grid(True,ls='--')
    plt.ylim([0,1])
    plt.xlim([-1,2])
    plt.legend(loc='best')
    plt.show()