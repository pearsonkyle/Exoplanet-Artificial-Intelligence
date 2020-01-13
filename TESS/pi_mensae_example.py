from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def phase_bin(time,flux,per,tmid=0,cadence=16,offset=0.25):
    '''
        Phase fold data and bin according to time cadence
        time - [days]
        flux - arbitrary unit
        per - period in [days]
        tmid - value in days
        cadence - spacing to bin data to [minutes] 
    '''
    phase = ((time-tmid)/per + offset)%1

    sortidx = np.argsort(phase)
    sortflux = flux[sortidx]
    sortphase = phase[sortidx]

    cad = cadence/60./24/per # phase cadence according to kepler cadence
    pbins = np.arange(0,1+cad,cad) # phase bins
    bindata = np.zeros(pbins.shape[0]-1)
    for i in range(pbins.shape[0]-1):
        pidx = (sortphase > pbins[i]) & (sortphase < pbins[i+1])

        if pidx.sum() == 0 or np.isnan(sortflux[pidx]).all():
            bindata[i] = np.nan
            continue

        bindata[i] = np.nanmean(sortflux[pidx])

    phases = pbins[:-1]+np.diff(pbins)*0.5

    # remove nans
    #nonans = ~np.isnan(bindata)
    #return phases[nonans],bindata[nonans]
    return phases, bindata

if __name__ == "__main__":

    lcfile="hlsp_tess-data-alerts_tess_phot_00261136679-s01_tess_v1_lc.fits"
    tpfile="hlsp_tess-data-alerts_tess_phot_00261136679-s01_tess_v1_tp.fits"
    
    fits.info(tpfile)
    tphdu=fits.open(tpfile)

    #Plot the first image of the FLUX Column with WCS overlay.
    tpf_data=tphdu[1].data
    first_image=tpf_data['FLUX'][1]

    #The aperture extension header contains the same WCS as that in the Pixels extension.
    wcs=WCS(tphdu[2].header)

    #Plot
    fig = plt.figure(figsize=(6,6))
    fig.add_subplot(111, projection=wcs)
    plt.imshow(first_image, origin='lower', cmap=plt.cm.binary)
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.grid(axis='both',ls='--')#,color='white', ls='solid')

    # extract the aperture 
    ap_image=(tphdu[2].data)
    ap_want=np.bitwise_and(ap_image, 2) / float(2)
    plt.imshow(ap_want, alpha=0.5)
    plt.show(block=True)

    # test an increased aperture size 
    ap_want2 = binary_dilation(ap_want).astype(int)

    #define a simple aperture function to sum-up specified pixels for one cadence
    def aperture_phot(image,aperture):
        return np.sum(image[aperture==1])
    
    #Use the map lambda functions to apply that function to all cadences
    opap_flux=np.array(list(map(lambda x: aperture_phot(x,ap_want), tpf_data['FLUX'])))
    opap_flux2=np.array(list(map(lambda x: aperture_phot(x,ap_want2), tpf_data['FLUX'])))

    # extract the time 
    time_bjd=tpf_data["TIME"]
    tpf_head=tphdu[1].header
    bjd_ref=tpf_head['BJDREFI'] + tpf_head['BJDREFF']
    time=time_bjd+bjd_ref

    # plot some data 
    plt.figure(figsize=(13,4))
    plt.plot(time_bjd,opap_flux,'.',label='Optimal Aperture',ms=6)
    plt.plot(time_bjd,opap_flux2,'.',label='Optimal Aperture + 1',ms=6)
    plt.legend(loc="lower left")
    plt.xlabel('Time (BJD)')
    plt.ylabel('Flux (e-/s)')
    plt.title("TESS Lightcurve Derived from Calibrated Pixels.")
    #plt.ylim([1340000,1450000])
    plt.show(block=True)

    # create a background flux time series (these values were already subtracted from the data)
    back_im=tpf_data['FLUX_BKG']
    back_opap_flux=list(map(lambda x: aperture_phot(x,ap_want), back_im))
    back_opap_flux2=list(map(lambda x: aperture_phot(x,ap_want2), back_im))
    
    # plot background vs time 
    plt.figure(figsize=(13,4))
    plt.plot(time_bjd,back_opap_flux,'.',label='Background in Optimal Aperture',ms=6)
    plt.plot(time_bjd,back_opap_flux2,'.',label='Background in Optimal Aperture+1',ms=6)
    plt.legend(loc="lower left")
    plt.xlabel('Time (BJD)',fontsize=14)
    plt.ylabel('Background Flux (e-/s)',fontsize=14)
    plt.title("TESS Background Lightcurve from Target Pixel File.",fontsize=14)
    plt.show(block=True)

    # time to compare to light curve files 
    lchdu=fits.open(lcfile)
    fits.info(lcfile)
    lcdata=lchdu[1].data
    sapflux=lcdata['SAP_FLUX']
    pdcflux=lcdata['PDCSAP_FLUX']
    quality=lcdata['QUALITY']
    time=lcdata['TIME']

    # remove nans
    nanmask = np.isnan(time) | np.isnan(pdcflux) 
    time = time[~nanmask]
    pdcflux = pdcflux[~nanmask]
    sapflux = sapflux[~nanmask]
    quality = quality[~nanmask]
    opap_flux = opap_flux[~nanmask]
    opap_flux2 = opap_flux2[~nanmask]

    '''
    Bit 1. Attitude Tweak
    Bit 2. Safe Mode
    Bit 3. Coarse Point
    Bit 4. Earth Point
    Bit 5. Argabrightening Event (Sudden brightening across the CCD.)
    Bit 6. Reaction Wheel Desaturation
    Bit 8. Manual Exclude
    Bit 10. Impulsive outlier
    Bit 12. Straylight detected
    '''
    bad_bits=np.array([1,2,3,4,5,6,8,10,12])
    value=0
    for v in bad_bits:
        value=value+2**(v-1)
        
    bad_data=np.bitwise_and(quality, value) >= 1 
    bad_data = bad_data | (pdcflux==0) | np.isnan(pdcflux)

    # see which data was removed with bad pixels
    plt.plot(time[~bad_data],pdcflux[~bad_data],'.',label='PDC')
    plt.plot(time[bad_data],pdcflux[bad_data],'.',label='BAD')
    plt.legend(loc='lower left')
    plt.xlabel('TIME (BTJD)')
    #plt.ylim([1432000, 1442000])
    plt.show(block=True)

    # check out the momentum data 
    fluxcent_col=lcdata['MOM_CENTR1'][~nanmask]
    fluxcent_row=lcdata['MOM_CENTR2'][~nanmask]
    distance=((fluxcent_col-np.nanmean(fluxcent_col))**2 + (fluxcent_row-np.nanmean(fluxcent_row))**2)**(0.5)

    mom_dump=np.bitwise_and(quality, 2**5) >= 1 
    mom_bad = np.zeros(mom_dump.shape[0])
    # loop through time values and remove data surrounding momentum dump
    for i in range(mom_dump.shape[0]):
        if mom_dump[i] == 1:
            bmask = (time > time[i]-0.5/24) & (time < time[i]+0.5/24)
            mom_bad[bmask] = 1
    mom_bad = mom_bad.astype(bool)

    # add data surrounding momentum dump to bad pixels
    bad_data = bad_data | mom_bad | ((time>1347) & (time<1350))

    plt.plot(time[~bad_data], pdcflux[~bad_data]/np.mean(pdcflux[~bad_data]), '.',label='PDC')
    plt.plot(time[mom_bad], pdcflux[mom_bad]/np.mean(pdcflux[~bad_data]), '.', label= 'bad mom' )
    plt.vlines(time[mom_dump],0.99,1.01,colors='r',label="Momentum Dump")
    plt.xlabel('TIME(BTKD)')
    plt.legend(loc="upper left")
    plt.show(block=True)
    
    # check for change in relative flux vs position 
    # also subtract off bad pixels 
    dflux = pdcflux[~bad_data]/np.nanmedian(pdcflux[~bad_data])
    dflux2 = opap_flux2[~bad_data]/np.nanmedian(opap_flux2[~bad_data])
    dy = fluxcent_row[~bad_data]-np.nanmedian(fluxcent_row[~bad_data])
    dx = fluxcent_col[~bad_data]-np.nanmedian(fluxcent_col[~bad_data])
    t = time[~bad_data]
    
    dist = np.sqrt(dx**2 + dy**2) 

    # create a phase folded light curve 
    pars = {'per': 6.266571, 'tm': 1325.503694, 'rp': 0.014312, 'ar': 15.9187, 'u1': 0.4242, 'u2': 0.1400, 'inc': 89.325, 'ecc': 0, 'ome': 0, 'a0': 1, 'a1': 0, 'a2': 0}

    # create phase folded data 
    phase = ((t-pars['tm'])/pars['per']+0.25 )%1

    # perform a sigma clip 
    sidx = np.argsort(phase)
    sf = median_filter(dflux[sidx], 16 ) # int(f.shape[0]/100) )
    res = dflux[sidx]-sf
    std = np.nanstd( res ) 
    smask = np.abs(res) < 3*std

    sf2 = median_filter(dflux2[sidx], 16 ) # int(f.shape[0]/100) )
    res = dflux2[sidx]-sf2
    std2 = np.nanstd( res ) 
    smask2 = np.abs(res) < 3*std2

    # create a mosaic 
    f = plt.figure( figsize=(10,8) )
    plt.subplots_adjust(top=0.95, bottom=0.09, left=0.1, right=0.98, wspace=0)
    ax = [
        plt.subplot2grid( (3,3), (0,0), colspan=3 ),
        plt.subplot2grid( (3,3), (1,0), colspan=3 ),
        
        plt.subplot2grid( (3,3), (2,0) ),
        plt.subplot2grid( (3,3), (2,1) ),
        plt.subplot2grid( (3,3), (2,2) )
    ]

    ax[0].plot(t[sidx][smask],dflux[sidx][smask],'k.') 
    #ax[0].plot(t[sidx][smask],sf[smask],'r.')
    ax[0].set_xlabel('Time (day)')
    ax[0].set_ylabel('Relative Flux')

    ax[1].plot(t[sidx][smask],dist[sidx][smask],'k.' )
    #ax[1].plot(t[sidx][smask2],dflux2[sidx][smask2],'ko') 
    #ax[1].plot(t[sidx][smask2],sf2[smask2],'r.')
    ax[1].set_xlabel('Time (day)')
    ax[1].set_ylabel('Pixel Offset')


    ax[2].plot(dy, (dflux-1)*1e6, 'k.',alpha=0.5)
    ax[2].set_ylabel('Relative Flux Change (ppm)')
    ax[2].set_xlabel('dy (px)')
    ax[2].set_ylim([-2000,2000])

    ax[3].plot(dx, (dflux-1)*1e6, 'k.',alpha=0.5)
    #ax[3].yaxis.label.set_visible(False)
    ax[3].set_yticks([])
    ax[3].set_xlabel('dx (px)')
    ax[3].set_ylim([-2000,2000])

    ax[4].plot(dist, (dflux-1)*1e6, 'k.',alpha=0.5)
    #ax[4].yaxis.label.set_visible(False)
    ax[4].set_yticks([])   
    ax[4].set_xlabel('distance (px)')
    ax[4].set_ylim([-2000,2000])

    plt.show(block=True)


    # create a mosaic 
    f = plt.figure( figsize=(10,8) )
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.1, right=0.98, wspace=0, hspace=0.34)
    ax = [
        plt.subplot2grid( (4,2), (0,0), colspan=2 ),
        plt.subplot2grid( (4,2), (1,0), colspan=2 ),
        plt.subplot2grid( (4,2), (2,0), colspan=2 ),

        plt.subplot2grid( (4,2), (3,0), ),
        plt.subplot2grid( (4,2), (3,1), ),
    ]
    ax[0].set_title('Pi Mensae b Aperture Selection')
    ax[0].plot(t[sidx][smask],dflux[sidx][smask],'k.',label='SPOC') 
    ax[0].set_xlabel('Time (day)')
    ax[0].set_ylabel('Relative Flux')
    ax[0].set_ylim([0.999,1.001])
    ax[0].set_xlim([1325.31,1347])
    ax[0].legend(loc='best')
    ax[0].grid(True,ls='--')
    ax[0].vlines(time[mom_dump],0.999,1.001,color='g',linestyle='--')

    #ax[1].plot(t[sidx][smask],dist[sidx][smask],'k.' )
    ax[2].plot(t[sidx][smask],dx[sidx][smask],'r.',alpha=0.5,label='dx' )
    ax[2].plot(t[sidx][smask],dy[sidx][smask],'b.',alpha=0.5,label='dy' )
    ax[2].vlines(time[mom_dump],-0.1,0.1,color='g',linestyle='--')
    ax[2].set_xlabel('Time (day)')
    ax[2].set_ylabel('Pixel Offset')
    ax[2].set_xlim([1325.31,1347])
    ax[2].legend(loc='best')
    ax[2].grid(True,ls='--')

    ax[1].plot(t[sidx][smask2],dflux2[sidx][smask2],'k.',label='SPOC+1',alpha=0.5) 
    ax[1].set_xlabel('Time (day)')
    ax[1].set_ylabel('Relative Flux')
    ax[1].legend(loc='best')
    ax[1].set_ylim([0.999,1.001])
    ax[1].set_xlim([1325.31,1347])
    ax[1].legend(loc='best')
    ax[1].grid(True,ls='--')
    ax[1].vlines(time[mom_dump],0.999,1.001,color='g',linestyle='--')

    ax[3].plot(phase[sidx][smask], dflux[sidx][smask], 'k.',label='SPOC')
    ax[3].set_xlim([0.2,0.3])
    ax[3].set_ylim([0.999,1.001])
    ax[3].legend(loc='best')
    ax[3].set_xlabel('Phase')
    ax[3].set_ylabel('Relative Flux')
    ax[3].grid(True,ls='--')

    ax[4].plot(phase[sidx][smask2],dflux2[sidx][smask2], 'k.',alpha=0.5,label='SPOC+1') 
    ax[4].legend(loc='best')
    #ax[4].get_yaxis().set_visible(False)
    ax[4].set_xlim([0.2,0.3])
    ax[4].set_ylim([0.999,1.001])
    ax[4].set_xlabel('Phase')
    ax[4].grid(True,ls='--')
    ax[4].set_yticklabels([])

    plt.show(block=True)


    # phase folded fit
    #plt.scatter(phase[sidx][smask], dflux[sidx][smask], c=dist[sidx][smask])
    #plt.scatter(phase[sidx][smask2], 1000e-6+ dflux2[sidx][smask2], c=dist[sidx][smask2],alpha=0.5)
    #plt.xlabel('phase'); plt.ylabel('Flux'); plt.show(block=True)
    #plt.show(block=True)
    bp, bf = phase_bin(t[sidx][smask],dflux[sidx][smask],pars['per'],pars['tm'],cadence=4)

    plt.plot(phase[sidx][smask], dflux[sidx][smask], 'k.')
    plt.plot(phase[sidx][smask2], 1000e-6+ dflux2[sidx][smask2], 'r.',alpha=0.5)
    plt.plot(bp,bf,'go')
    plt.xlabel('phase'); plt.ylabel('Flux'); plt.show(block=True)
    plt.show(block=True)

    # fit a light curve model to the phase folded at
    pmask = (phase[sidx][smask] > 0.21) & (phase[sidx][smask] < 0.29)
    tp = phase[sidx][smask][pmask]*pars['per']
    fp = dflux[sidx][smask][pmask]
    
    pmask = (phase[sidx][smask2] > 0.21) & (phase[sidx][smask2] < 0.29)
    tp2 = phase[sidx][smask2][pmask]*pars['per']
    fp2 = dflux2[sidx][smask2][pmask]

    # Evaluate transit probability with CNN
    from tensorflow.keras.models import load_model
    from sklearn import preprocessing
    from cnn_train import make_cnn
    cnn = make_cnn(180)
    cnn.load_weights('tess_cnn1d.h5')
    pmask2 = (bp>0.21) & (bp<0.29)
    f180 = interp1d(bp[pmask2], bf[pmask2])
    xnew = np.linspace( min(bp[pmask2]), max(bp[pmask2]), 180)
    ynew = preprocessing.scale( f180(xnew) )
    prob = cnn.predict(ynew.reshape(1,-1,1))[0][0]
    print('transit probability: ',prob)

    # fit the phase folded data
    pars['tm'] = pars['per']*0.25
    from ELCA import lc_fitter
    # only report params with bounds, all others will be fixed to initial value
    mybounds = {
              'rp':[0,1],
              'tm':[min(tp),max(tp)],
              'a0':[-np.inf,np.inf],
              'a1':[-np.inf,np.inf],
              'a2':[-np.inf,np.inf]
              } 

    myfit = lc_fitter(
        tp,fp,
        dataerr=0.5*np.ones(fp.shape[0])*np.std(fp),
        init=pars,
        bounds= mybounds,
        nested=True
    )

    for k in myfit.data['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit.data['NS']['parameters'][k],myfit.data['NS']['errors'][k]) )


    myfit2 = lc_fitter(
        tp2,fp2,
        dataerr=0.5*np.ones(fp2.shape[0])*np.std(fp2),
        init=pars,
        bounds= mybounds,
        nested=True
    )

    for k in myfit2.data['freekeys']:
        print( '{}: {:.6f} +- {:.6f}'.format(k,myfit2.data['NS']['parameters'][k],myfit2.data['NS']['errors'][k]) )

    myfit.plot_results(show=True,t='NS')

'''
OG aperture 
    ln(ev)=  -6054.8526473234533      +/-  0.44548279963420856
    Total Likelihood Evaluations:        54619
    analysing data from chains/1-.txt
    rp: 0.014382 +- 0.000185

NEW aperture
     ln(ev)=  -4510.7134648063638      +/-  0.44261512148531706
    Total Likelihood Evaluations:        40531
    analysing data from chains/1-.txt
    rp: 0.016073 +- 0.000118

Paper Values ()
    0.01703 +- 0.00023
'''


# phase folded plot 
#plt.plot(phase[sidx][smask], sf[smask], 'k.',alpha=0.5)
#plt.plot(phase[sidx][smask2], sf2[smask2], 'r.',alpha=0.5)
#plt.xlabel('phase'); plt.ylabel('Flux'); plt.show(block=True)

