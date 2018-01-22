from graph_sensitivity import load_data,wavy,model_predict
from generate_data import dataGenerator

from scipy.interpolate import interp1d
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle

from time import sleep

def generate(N):

    try:
        X_test,y_test,pvals,keys,time = load_data('test_dt{}.pkl'.format(N),whiten=True,NULL=False)
    except:
        print('Generating Data:',N)
        settings = { 'ws':360, 'dt':360./N}
        hw = (0.5*N*settings['dt'])/60./24. # half width in days
        t = np.linspace( 1-hw,1+hw,N )
        dt = t.max()-t.min()

        # default transit parameters
        init = { 'rp':0.1, 'ar':12.0,     # Rp/Rs, a/Rs
                 'per':2, 'inc':90,         # Period (days), Inclination
                 'u1': 0.5, 'u2': 0,        # limb darkening (linear, quadratic)
                 'ecc':0, 'ome':0,          # Eccentricity, Arg of periastron
                 'a0':1, 'a1':0,            # Airmass extinction terms
                 'a2':0, 'tm':1 }           # tm = Mid Transit time (Days)

        pgrid_test = {
            # TEST data
            'rp': (np.array([200,500,1000,2500,5000,10000])/1e6)**0.5, # transit depth (ppm) -> Rp/Rs
            'per':np.linspace(*[2,4,5]),
            'inc':np.array([86,87,90]),
            'sig_tol':np.linspace(*[0.25,3,12]), # generate noise based on X times the tdepth

            # stellar variability systematics
            'phi':np.linspace(*[0,np.pi,4]),
            'A': np.array([250,500,1000,2000])/1e6,
            'w': np.array([6,12,24])/24., # periods in days
            'PA': [-4*dt,4*dt,100], # doubles amp, zeros amp between min time and max time, 1000=no amplitude change
            'Pw': [-12*dt,4*dt,100], #-12dt=period halfs, 4dt=period doubles, 1000=no change
        }

        data = dataGenerator(**{'pgrid':pgrid_test,'settings':settings,'init':init})
        data.generate()

        pickle.dump({'keys':data.keys,'results':data.results,'time':data.t}, open('pickle_data/test_dt{}.pkl'.format(N),'wb'))
        X_test,y_test,pvals,keys,time = load_data('test_dt{}.pkl'.format(N),whiten=True,NULL=False)

    return X_test,y_test,pvals,keys,time

if __name__ == "__main__":

    models = {
        'SVM': load_model('models/SVM_transit.h5'),
        'MLP': load_model('models/MLP_transit.h5'),
        'Wavelet MLP': load_model('models/Wavelet MLP_transit.h5'),
        'CNN 1D': load_model('models/CNN 1D_transit.h5'),
    }

    # load OG data
    X_test,y_test,pvals,keys,OGtime = load_data('transit_data_test.pkl',whiten=True,NULL=False)

    colors = {
        'MLP':'red',
        'Wavelet MLP':'green',
        'CNN 1D':'blue',
        'SVM': 'orange'
    }


    # resolution of data relative original
    npts = np.arange(X_test.shape[1]*0.25,X_test.shape[1]*2,30).astype(int)
    idxs = np.arange(X_test.shape[1])

    # filter by
    sig_idx = np.array(keys) == 'sig_tol'
    sigs = np.unique(pvals[:,sig_idx])
    sidx = pvals[:,sig_idx].flatten()>1
    X_test = X_test[sidx]

    # allocate grid
    acc_grid = {};
    for k in models.keys():
        acc_grid[k] = []

    for n in npts:

        X_low,y_test,pvals,keys,time = generate(n)
        sidx = pvals[:,sig_idx].flatten()>1
        X_low = X_low[sidx]

        for i in range(X_test.shape[0]):
            f = interp1d(time,X_low[i],kind='linear',bounds_error=False,fill_value='extrapolate')

            # interpolate data back to original grid
            X_test[i] = f(OGtime)

        # loop through all models and compute accuracy
        for k in models.keys():
            if k == 'Wavelet MLP':
                X_w = wavy(X_test)
                acc_grid[k].append( model_predict(models[k], X_w) )
            elif k == "CNN 1D":
                X_c = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
                acc_grid[k].append( model_predict(models[k], X_c) )
            else:
                acc_grid[k].append( model_predict(models[k], X_test) )

    #colors = cm.rainbow( np.linspace(0,1,len(models.keys())))
    f,ax = plt.subplots(1)
    for k in models.keys():
        ax.plot( 100*(npts/X_test.shape[1]), acc_grid[k],'-',label=k,c=colors[k] )

    ax.set_title('Transit Detection After Interpolation')
    ax.set_xlabel('Data Resolution (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='best')
    #ax.set_ylim([88,100])
    #ax.set_xlim([20,100])
    plt.show()
