import numpy as np
import matplotlib.pyplot as plt

wave = lambda t,A,w,phi : A*np.sin(2*np.pi*t/w + phi)
wave_amp = lambda t,A,w,wA,phi : ( A+A*np.sin(2*np.pi*t/wA) )*np.sin(2*np.pi*t/w + phi)
wave_freq = lambda t,A,w,ww,phi : A*np.sin(2*np.pi*t/(w+w*np.sin(2*np.pi*t/ww)) + phi)
wave_quasi = lambda t,A,w,wA,ww,phi : ( A+A*np.sin(2*np.pi*t/wA) )*np.sin(2*np.pi*t/(w+w*np.sin(2*np.pi*t/ww)) + phi)

qwave = lambda t,pars : ( pars['A']+pars['A']*np.sin(2*np.pi*t/pars['PA']) )*np.sin(2*np.pi*t/(pars['w']+pars['w']*np.sin(2*np.pi*t/pars['Pw'])) + pars['phi'])


if __name__ == "__main__":
    t = np.linspace(0.875,1.125,180) # [days]

    changes = {'A':[0.5,1,2], 'PA':[-1,1,100],'w':[6./24,12./24,24./24],'Pw':[-3,1,100],'phi':[0,np.pi/4,np.pi/2]}

    label = {'A':'A', 'PA':r'$P_A$','w':r'$\omega$','Pw':r'$P_\omega$','phi':r'$\phi$'}
    ncols = 2
    nkeys = len(changes.keys())+1
    f,ax = plt.subplots(int(nkeys/ncols),ncols,figsize=(9,5))
    plt.subplots_adjust(left=0.09,bottom=0.09,right=0.98,top=0.94,wspace=0.04,hspace=0.04)
    f.suptitle('Variability Shape Analysis')

    for k in range(nkeys-1):
        i = int(k/ncols)
        j = k%ncols
        key = list(changes.keys())[k]
        init = {'A':1,'PA':1000,'w':6./24,'Pw':1000,'phi':0}

        for v in changes[key]:
            init[key] = v
            ax[i,j].plot(t, qwave(t-t[0], init),label="{} = {:.2f}".format(label[key], v))

        ax[i,j].legend(loc='best')
        ax[i,j].set_xlabel('Time (Days)')
        #ax[i,j].set_ylim([0.986,1.0005])

        if j == 1:
            ax[i,j].get_yaxis().set_ticks([])
        else:
            ax[i,j].set_ylabel('Relative Flux')

    f.delaxes(ax[-1,-1])
    plt.show()
