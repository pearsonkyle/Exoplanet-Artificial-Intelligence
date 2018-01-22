from graph_sensitivity import load_data,wavy
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from keras.utils import np_utils
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
    X_test,y_test,pvals,keys,time = load_data('transit_data_test.pkl',whiten=True)
    Xw_test = wavy(X_test)
    Xc_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    #y_test = np_utils.to_categorical(y_test, num_classes=2)

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
        'SVM': 'orange',
        'BLS': 'cyan'
    }

    f,ax = plt.subplots(1)

    # BLS ROC SCORE
    data = pickle.load(open('pickle_data/LS_predictions.pkl','rb'))
    sig_tol = 1./ (np.array(data['rps']) / (np.array(data['stds'])/1e6 ))
    y_predict = np.abs(np.array(data['mids'])/90.) * -0.25*np.sign(data['rps'])*sig_tol
    fpr, tpr, threshold_list_ls = roc_curve(data['y_test'], y_predict)
    roc_auc = auc(fpr, tpr)
    ax.plot( fpr,tpr, label="{:.2f} - {}".format(roc_auc,'BLS'),color=colors['BLS'] )

    # compute the ROC plot
    for k in models.keys():
        y_pred = models[k].predict(tests[k])
        
        fpr, tpr, threshold_list = roc_curve(y_test,y_pred)
        roc_auc = auc(fpr, tpr)

        ax.plot( fpr,tpr, label="{:.2f} - {}".format(roc_auc,k),color=colors[k] )


    ax.plot([0,1],[0,1],'k--')
    ax.legend(loc='best')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    #ax.set_ylim([0,101])
    #ax.set_xlim([0,2.01])
    plt.show()
