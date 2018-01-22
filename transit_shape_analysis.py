
from ELCA import transit
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    t = np.linspace(0.875,1.125,180)

    changes = { 'rp':[0.05,0.075,0.1],'ar':[10,12,14], 'ecc':[0,0.25,0.5],'inc':[90,87,86],'per':[2,3,4],'u1':[0.3,0.5,0.7] }
    label = {'rp':r'$R_p/R_s$','ar':r'$a/R_s$','ecc':r'$e$','inc':r'$i$','per':'P','u1':r'$u_1$'}
    ncols = 2
    nkeys = len(changes.keys())
    f,ax = plt.subplots(int(nkeys/ncols),ncols,figsize=(9,5))
    plt.subplots_adjust(left=0.09,bottom=0.09,right=0.98,top=0.94,wspace=0.04,hspace=0.04)
    f.suptitle('Transit Shape Analysis')
    for k in range(nkeys):
        i = int(k/ncols)
        j = k%ncols
        key = list(changes.keys())[k]
        init = { 'rp':0.1, 'ar':12,       # Rp/Rs, a/Rs
                     'per':2, 'inc':90, # Period (days), Inclination
                     'u1': 0.5, 'u2': 0,          # limb darkening (linear, quadratic)
                     'ecc':0, 'ome':0,            # Eccentricity, Arg of periastron
                     'a0':1, 'a1':0,              # Airmass extinction terms
                     'a2':0, 'tm':1 }          # tm = Mid Transit time (Days)

        for v in changes[key]:
            init[key] = v
            ax[i,j].plot(t, transit(time=t, values=init),label="{} = {}".format(label[key], v))

        ax[i,j].legend(loc='best')
        ax[i,j].set_xlabel('Time (Days)')
        ax[i,j].set_ylim([0.986,1.0005])

        if j == 1:
            ax[i,j].get_yaxis().set_ticks([])
        else:
            ax[i,j].set_ylabel('Relative Flux')

    plt.show()
