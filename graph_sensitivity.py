import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from matplotlib.pyplot import cm
from sklearn import preprocessing
import pywt


arr = lambda x : np.array( list(x),dtype=np.float )
def load_data(fname='transit_data.pkl',categorical=False,whiten=True,DIR='pickle_data/',NULL=True):

    data = pickle.load(open(DIR+fname,'rb'))

    # convert to numpy array fo float type from object type
    pvals = arr(data['results'][:,0])
    transits = arr(data['results'][:,1])
    null = arr(data['results'][:,2])

    if NULL:
        X = np.vstack([transits,null])
        y = np.hstack([np.ones(transits.shape[0]), np.zeros(null.shape[0])] )
    else:
        X = np.vstack([transits])
        y = np.hstack([np.ones(transits.shape[0])] )

    if categorical: y = np_utils.to_categorical(y, np.unique(y).shape[0] )
    if whiten: X = preprocessing.scale(X,axis=1)

    return X,y,pvals,data['keys'],data['time']

def summary(pval,keys):
    # print out pvals with keys
    for i in range(0,len(pval)):
        print('{}: {:.2f}'.format(keys[i],pval[i]) )

def model_predict(model,X):
    pred = model.predict(X)
    return 100*pred.sum()/pred.shape[0]

# db1, db2,
def wavy(Xt,wavelet='db2'):
    cA,cD = pywt.dwt(Xt[0],wavelet,'symmetric')
    size = cA.shape[0]+cD.shape[0]

    X = np.zeros((Xt.shape[0],size))
    for i in range(Xt.shape[0]):
        cA, cD = pywt.dwt(Xt[i], wavelet)
        X[i] = list(cA) + list(cD)
    return X

# RAINBOW SPECTRUM!
def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0, 270, 270 / num_colors):
        hue = (i+90)/360.
        lightness = (50)/100.
        saturation = (100)/100.
        colors.append(hls_to_rgb(hue, lightness, saturation))
    return colors


if __name__ == "__main__":

    models = {
        'SVM': load_model('models/SVM_transit.h5'),
        'MLP': load_model('models/MLP_transit.h5'),
        'Wavelet MLP': load_model('models/Wavelet MLP_transit.h5'),
        'CNN 1D': load_model('models/CNN 1D_transit.h5'),
    }

    # load DATA
    X_test,y_test,pvals,keys,time = load_data('transit_data_test.pkl',whiten=True,NULL=False)
    Xw_test = wavy(X_test)
    Xc_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

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

    # find all the unique transit depths and sigmas
    rp_idx = np.array(keys) == 'rp'
    sig_idx = np.array(keys) == 'sig_tol'
    rps = np.unique(pvals[:,rp_idx])
    sigs = np.unique(pvals[:,sig_idx])

    # allocate grid
    acc_grid = {}
    for k in models.keys():
        acc_grid[k] = np.zeros( (rps.shape[0],sigs.shape[0]) )

    for i in range(rps.shape[0]):
        for j in range(sigs.shape[0]):
            ridx = pvals[:,rp_idx].flatten()==rps[i]
            sidx = pvals[:,sig_idx].flatten()==sigs[j]

            # loop through all models and compute accuracy
            for k in models.keys():
                acc_grid[k][i,j] = model_predict(models[k],tests[k][ridx&sidx])

    f,ax = plt.subplots(1)
    i = 0;
    for k in models.keys():
        ax.plot( np.round(sigs,2), acc_grid[k].mean(0),label=k,c=colors[k] )
        i += 1

    ax.legend(loc='best')
    ax.set_xlabel('Transit Depth / Noise Scatter')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Average Transit Detection Accuracy')
    ax.set_ylim([0,101])
    ax.set_xlim([0,2.01])
    plt.show()

    '''
    # PREDICTED TESS ACCURACY PLOT - run after above
    # Accuracy for apparent magnitude vs. planet size for (solar type star)
    from scipy.interpolate import interp1d
    Re = 6371 # km
    Rs = 695700 # km
    facc = interp1d( np.round(sigs,2), acc_grid['CNN 1D'].mean(0),fill_value='extrapolate' )

    # factors of earth radii (0.7 -> 20)
    R1 = np.logspace(-0.15,1.3,75)
    R2 = np.linspace(0.75,10,75)
    R = 0.5*(R1+R2)
    depth = 1e6*(R*Re / Rs) **2

    # Figure 8, Ricker 2014, photometric precision and magnitude
    mag = [6,7,8,9,10,11,12,13,14,15,16,18]
    sig = [60,70,90,150,200,300,500,1000,2000,6000,10000,70000]
    fphot = interp1d( mag,sig,kind='linear' )
    mags = np.linspace(6,16,75)
    sigss = fphot(mags) # returns photometric precision sigma ppm

    accs = np.zeros( (R.shape[0],mags.shape[0]) )
    for i in range(R.shape[0]):
        for j in range(mags.shape[0]):
            accs[i,j] = facc( depth[i]/sigss[j] )

    plt.imshow(accs,interpolation='none',cmap='Spectral',vmin=0,vmax=100,origin='lower' )
    plt.xticks(np.arange(mags.shape[0])[::10],np.round(mags,0)[::10],rotation='vertical')
    plt.yticks(np.arange(depth.shape[0])[::10],  np.round( (depth[::10]/1e6)**0.5*Rs/Re,1)  )
    plt.xlabel('Stellar Magnitude')
    plt.ylabel('Planet Radius (Earth radii)')
    plt.title('CNN Detection Accuracy given TESS Noise Statistics')
    cbar = plt.colorbar()
    cbar.set_label('Accuracy (%)')
    plt.show()
    '''
