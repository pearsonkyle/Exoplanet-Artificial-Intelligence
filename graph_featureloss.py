from graph_sensitivity import load_data,wavy,model_predict
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle

if __name__ == "__main__":

    models = {
        'SVM': load_model('models/SVM_transit.h5'),
        'MLP': load_model('models/MLP_transit.h5'),
        'Wavelet MLP': load_model('models/Wavelet MLP_transit.h5'),
        'CNN 1D': load_model('models/CNN 1D_transit.h5'),
    }

    # load DATA
    X_test,y_test,pvals,keys,time = load_data('transit_data_test.pkl',whiten=True,NULL=False)
    #Xw_test = wavy(X_test)
    #Xc_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

    tests = {
        'MLP':X_test,
        'Wavelet MLP':X_test,
        'CNN 1D':X_test,
        'SVM':X_test
    }

    colors = {
        'MLP':'red',
        'Wavelet MLP':'green',
        'CNN 1D':'blue',
        'SVM': 'orange'
    }


    npts = np.arange(0,X_test.shape[1]*0.55,5).astype(int)
    idxs = np.arange(X_test.shape[1])

    sig_idx = np.array(keys) == 'sig_tol'
    sigs = np.unique(pvals[:,sig_idx])

    # For N, 0->50% of the features, select C=RandomInt(1,N/10) chunks to take out of the data where the chunk size is equal to N/C. Then random positions within the lightcurve as zeroed out with no overlapping chunks?

    # allocate grid
    acc_grid = {}
    for k in models.keys():
        acc_grid[k] = []



    accuracy = []
    for n in npts:
        print('n:',n)
        sidx = pvals[:,sig_idx].flatten()>1
        X_r = np.copy( tests[k][sidx] )

        # random chunks to zero out
        for i in range(X_r.shape[0]):
            nchunks = np.random.randint(0,n/10+0.01) + 1
            chunksize = int(n/nchunks)
            ridxs = np.random.randint(0,X_r.shape[1],nchunks) #idxs:idxs+chunksize = 0

            # filter out over lapping chunks
            if chunksize > 1:
                ridxs.sort()
                diff = np.diff(ridxs)

                while (diff < chunksize).any():
                    ridxs = np.random.randint(0,X_r.shape[1],nchunks)
                    ridxs.sort()
                    diff = np.diff(ridxs)

            # zero out chunks
            for j in range(nchunks):
                X_r[i,ridxs[j]:ridxs[j]+chunksize] = 0

        # loop through all models and compute accuracy
        for k in models.keys():
            if k == 'Wavelet MLP':
                X_w = wavy(X_r)
                acc_grid[k].append( model_predict(models[k], X_w) )
            elif k == "CNN 1D":
                X_c = X_r.reshape((X_r.shape[0],X_r.shape[1],1))
                acc_grid[k].append( model_predict(models[k], X_c) )
            else:
                acc_grid[k].append( model_predict(models[k], X_r) )

    #colors = cm.rainbow( np.linspace(0,1,len(models.keys())))
    f,ax = plt.subplots(1)
    for k in models.keys():
        ax.plot( 100*(X_test.shape[1]-npts)/X_test.shape[1], acc_grid[k],label=k,c=colors[k] )

    ax.set_title('Transit Detection with Feature Loss')
    ax.set_xlabel('Number of Data Points (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='best')
    #ax.set_ylim([88,100])
    ax.set_xlim([25,100])
    plt.show()
