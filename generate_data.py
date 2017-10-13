import pickle
import numpy as np
from ELCA import transit
import multiprocessing as mp
from itertools import product
from sklearn import preprocessing

np.random.seed(1337)

class dataGenerator(object):
    def __init__(self,**kwargs):

        self.pgrid = kwargs['pgrid']
        self.settings = kwargs['settings']
        self.init = kwargs['init'] # initial values for transit

        # order of parameter keys
        self.keys = list(self.pgrid.keys())

        # generate time
        npts = self.settings['ws']/self.settings['dt']
        hw = (0.5*npts*self.settings['dt'])/60./24.
        self.t = np.linspace( 1-hw,1+hw,npts )


    def super_worker(self, *args):
        # generate parameter dict for transit function input
        pars = self.init.copy()
        for k in range(len(self.keys)):
            pars[self.keys[k]] = args[k]

        # compute white noise
        noise = pars['rp']**2 / pars['sig_tol']
        ndata = np.random.normal(1, noise, len(self.t))

        # generate transit + variability
        t = self.t - np.min(self.t)
        A = pars['A']+pars['A']*np.sin(2*np.pi*t/pars['PA'])
        w = pars['w']+pars['w']*np.sin(2*np.pi*t/pars['Pw'])
        data = transit(time=self.t, values=pars) * ndata * (1+A*np.sin(2*np.pi*t/w + pars['phi']))

        # add systematic into noise "same distribution different points"
        ndata = np.random.normal(1, noise, len(self.t)) * (1+A*np.sin(2*np.pi*t/w + pars['phi']))

        return args,data,ndata

    def generate(self,null=False):
        plist = [self.pgrid[k] for k in self.keys]
        pvals = list(product(*plist)) # compute cartesian product of N sets

        # evaluate all of the transits with multiprocessing
        pool = mp.Pool()
        self.results = np.array( pool.starmap(self.super_worker, pvals) )
        pool.close()
        pool.join()

        del plist
        del pvals
        # split up the results
        #self.pvals,self.transits,self.null = zip(*results)
        #self.pvals = results[:,0]

        
arr = lambda x : np.array( list(x),dtype=np.float )
def load_data(fname='transit_data_train.pkl',categorical=False,whiten=True,DIR='pickle_data/'):

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


if __name__ == "__main__":

    # Generate time data
    settings = { 'ws':360, 'dt':2 }
    # window size (ws/dt = num pts) (MINUTES)
    # time step (observation cadence) (MINUTES)
    npts = settings['ws']/settings['dt']
    hw = (0.5*npts*settings['dt'])/60./24. # half width in days
    t = np.linspace( 1-hw,1+hw,npts )
    dt = t.max()-t.min()


    # default transit parameters
    init = { 'rp':0.1, 'ar':12.0,     # Rp/Rs, a/Rs
             'per':2, 'inc':90,         # Period (days), Inclination
             'u1': 0.5, 'u2': 0,        # limb darkening (linear, quadratic)
             'ecc':0, 'ome':0,          # Eccentricity, Arg of periastron
             'a0':1, 'a1':0,            # Airmass extinction terms
             'a2':0, 'tm':1 }           # tm = Mid Transit time (Days)

    # training data
    pgrid = {
        'rp': (np.array([200,500,1000,2500,5000,10000])/1e6)**0.5, # transit depth (ppm) -> Rp/Rs

        'per':np.linspace(*[2,4,5]),
        'inc':np.array([86,87,90]),
        'sig_tol':np.linspace(*[1.5,4.5,4]), # generate noise based on X times the tdepth

        # stellar variability systematics
        'phi':np.linspace(*[0,np.pi,4]),
        'A': np.array([250,500,1000,2000])/1e6,
        'w': np.array([6,12,24])/24., # periods in days
        'PA': [-4*dt,4*dt,100], # doubles amp, zeros amp between min time and max time, 1000=no amplitude change
        'Pw': [-12*dt,4*dt,100], #-12dt=period halfs, 4dt=period doubles, 1000=no change
    }


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


    data = dataGenerator(**{'pgrid':pgrid,'settings':settings,'init':init})
    data.generate()

    pickle.dump({'keys':data.keys,'results':data.results,'time':data.t}, open('pickle_data/transit_data_train.pkl','wb'))

    print('number of samples:',len(data.results))
