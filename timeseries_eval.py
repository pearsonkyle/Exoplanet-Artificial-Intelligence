import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from ELCA import transit

def time_series_eval(data,mlp,spacing=5,ws=180,debug=False,delta=0.5):
    '''
        breaks data up into N lightcurves based on the windowsize and spacing
        evaluates the possibility that a transit is within each window using a
        mlp classifier in tensorflow
    '''

    # alloc array to store results
    ncurves = max(1,int( (data.shape[0]-ws)/spacing ))
    newdata = np.ones((ncurves,ws)) #180 is the window size for NN
    alphas = np.zeros(data.shape[0]) # colors on final plot
    #stds = np.zeros(data.shape[0])
    #newtime = []

    # remove outliers - phase binning with no data sets array to zero
    # convert zeros to different value
    data[data==0] = np.nan

    # create transits that overlap in time
    for i in range(ncurves):
        newdata[i] = data[i*spacing:ws+i*spacing] - np.nanmean(data[i*spacing:ws+i*spacing])
        newdata[i] /= np.nanstd(newdata[i])
        #newtime.append( t[i*spacing:ws+i*spacing].mean() )

    # convert nans to zero -> fill with random number
    newd = np.nan_to_num(newdata)
    #newd[newd==0] = np.random.normal(np.nanmean(newdata[newd!=0]),np.nanstd(newdata[newd!=0]),newd[newd==0].shape[0])

    try:
        timepredict = mlp.predict(newd,batch_size=1,verbose=0)
    except:
        # reshape for CNN
        newd = newd.reshape(newd.shape[0],newd.shape[1],1)
        timepredict = mlp.predict(newd,batch_size=1,verbose=0)

    # generate probability distribution in time based on transit predictions
    for i in range(ncurves):
        alphas[i*spacing:ws+i*spacing] += timepredict[i,0]
    alphas[(i+1)*spacing:ws+(i+1)*spacing] += timepredict[i,0]

    # normalize the alphas and scale up for plotting purposes
    alphas -= min(alphas)
    alphas /= max(alphas)#(0.25*ws/spacing)
    alphas = np.max([0.01+np.zeros(alphas.shape[0]),alphas],0)
    alphas = np.min([alphas,np.ones(alphas.shape[0])],0)

    # darken the
    rgba_colors = np.ones((data.shape[0],4))
    rgba_colors[:,:3] = 0.05 # blacken the color

    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas

    return alphas,rgba_colors

def plot_time_series(time,data,alphas,rgba_colors,depth=900,title='Time Series Transit Detection'):

    f = plt.figure(figsize=(11,4))
    ax = [plt.subplot2grid((3,7),(0,0),colspan=5,rowspan=2),
          plt.subplot2grid((3,7),(2,0),colspan=5),
          plt.subplot2grid((3,7),(0,5),colspan=2,rowspan=2),
          plt.subplot2grid((3,7),(2,5),colspan=2) ]


    #ax[0].set_ylim( [np.nanmean(data)-3*np.nanstd(data)-np.nanmean(data)*depth/1e6,np.nanmean(data)+3*np.nanstd(data)] )
    ax[0].set_xlim([min(time),max(time)])
    data = np.nan_to_num(data)
    ax[0].scatter(np.nan_to_num(time),np.nan_to_num(data), color=rgba_colors)
    #ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel('Relative Flux')
    ax[0].set_title(title)

    #for i in 250+np.arange(0,2000,500):
    #    ax[0].plot( [i,i],[0.9925,1.0075],'k--')

    ax[1].plot(alphas)
    #ax[1].set_xlim([0,data.shape[0]])
    #ax[1].set_ylim([0,1])
    ax[1].yaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[1].set_ylabel( 'Probability' )
    ax[1].set_xlabel( 'Time')

    '''
    fdata = np.copy( data[:int(data.shape[0]/500)*500].reshape(-1,500).mean(0) )
    alphas2,rgba_colors2 = time_series_eval(fdata,mlp,spacing=5,ws=180)
    ax[2].scatter(np.arange(len(fdata)),fdata, color=rgba_colors2)
    ax[2].set_xlim([0,fdata.shape[0]])
    ax[2].set_ylim([0.9925,1.0075])
    ax[2].xaxis.set_visible(False)
    ax[2].set_ylabel('Relative Flux')
    ax[2].set_title('Phase Folded')
    ax[2].yaxis.set_visible(False)

    ax[3].plot(alphas2)
    ax[3].set_xlim([0,fdata.shape[0]])
    ax[3].yaxis.set_ticklabels([])
    ax[3].xaxis.set_ticklabels([])
    ax[3].set_ylabel( 'Probability' )
    ax[3].set_xlabel( 'Phase')
    ax[3].yaxis.set_visible(False)

    ax[3].set_ylim([0,alphas2.max()])
    ax[1].set_ylim([0,alphas2.max()])
    '''

    plt.show()


if __name__ == "__main__":
    # load data
    #X,y,pvals,keys,time = load_data('transit_data.pkl',whiten=True)
    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

    # load models
    #mlp = load_model('models/NN_transit.h5')
    mlp = load_model('models/CNN 1D_transit.h5')
    
    # GENERATE DATA
    NDAYS = 30
    CADENCE = 29.4 # minutes
    NPTS = int(NDAYS*24*60/CADENCE)

    # transit data
    t = np.linspace(0.5, 0.5+NDAYS, NPTS)
    init = { 'rp':0.1, 'ar':12.0,     # Rp/Rs, a/Rs
             'per':8, 'inc':90,         # Period (days), Inclination
             'u1': 0.7, 'u2': 0,        # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,          # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,            # Airmass extinction terms
             'a2':0, 'tm':1 }           # tm = Mid Transit time (Days)

    model = transit(time=t,values=init)
    data = model * np.random.normal(1,1000e-6,NPTS) + 500e-6*np.sin(2*np.pi*np.arange(NPTS)/(0.5*NPTS) )


    alphas,rgba_colors = time_series_eval(data,mlp,spacing=5,ws=180)
    plot_time_series(t, data,alphas,rgba_colors)

    # TODO implement BLS