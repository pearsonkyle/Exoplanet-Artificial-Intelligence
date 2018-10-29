# Searching for Exoplanets using Artificial Intelligence

White Paper Link: https://academic.oup.com/mnras/article/doi/10.1093/mnras/stx2761/4564439/Searching-for-Exoplanets-using-Artificial 

arXiv: https://arxiv.org/abs/1706.04319

This project uses machine learning to search for transiting exoplanets in planetary search surveys. Multiple neural networks (MLP, CNN, Wavelet MLP) are trained to recognize patterns from artificial light curves that mimic real observations. The trained networks are validated with real data using the known ephemerii of transiting planets discovered from the [Kepler](https://www.nasa.gov/mission_pages/kepler/main/index.html) mission.

Dependencies: 
  * [Python3](https://www.continuum.io/downloads)
  * [Numpy](http://www.numpy.org/)
  * [Keras](https://keras.io/)
  * [TensorFlow](https://www.tensorflow.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Scikit-learn](http://scikit-learn.org/stable/)
  * [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
  * [Exoplanet Light Curve Analysis](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis)

## Getting started
It depends what you're interested in doing.. I would reccomend reading through the paper to get a general feel for what the code is doing. This particular machine learning algorithm is basically a glorified pattern recognition algorithm designed to recognize light curve shapes in photometric time series. I tried to include all of the relavant scripts that went into making this paper and their descriptions are below. If you have a question about anything please feel free to email me!


## Comprehensive File Guide 
`generate_data.py` - generates over 300,000 training and test samples from the parameter space grid in Table 1.

`quasiperiodicity.py` - Generates the variability shape analysis plot (Figure 4)

`transit_shape_analysis.py` - Generates various figures showing how the light curve shape changes as a function of the planet and orbit parameters. This plot is particularly useful when creating training data sets for exoplanets because not all parameters are going to yield an observable signal (careful! if you're unfamiliar with transiting exoplanet geometry you could easily bias your training data)

`model_fit_history` - creation of each neural network and its training. The training performance per epoch is saved and used to create Figure 6. 

`ROC_auc_score.py` - Creates the receiver operating characteristic plot to assess the accuracy of each algorithm. 

`graph_featureloss.py` - Explorations into the accuracy of the neural network if we're missing input data (Figure 11)

`graph_interpolate.py` - One of my favorite plots because it shows the accuracy of each algorithm after a signal has been interpolated from either a high or low resolution state. This is particularly useful to understand because it will allow one to apply this algorithm to an arbitrary transit survey without needing to retrain the network. Instead, just transform the input data to match what the network needs. Since we're dealing with a time sorted signal we can get away with this for the most part. Just make sure the transit signal is greater than a few data points. 

`graph_sensitivity.py` - Explores how the detection accuracy of each algorithm changes with signal to noise ratio. 

`timeseries_eval.py` - a simple script that will evaluate a time series light curve that is larger than the input for the neural network by breaking it up into smaller lightcurves. Think of it like a sliding box-car evaluation along the light curve. 


## Ways to improve the research 
This research was initially created for a class project and the field of machine learning is accelerating rapidly. I have a few suggestions on how to improve this research for future publications. Perhaps including some form of hyper-parameter optimization (e.g. [HyperOpt](http://hyperopt.github.io/hyperopt/)). The optimization of neural network architectures will be a standard in the future of machine learning algorithms. This is often something that is over looked in machine learning particularly when scientists (not computer scientists) use these algorithms. There is no doubt machine learning is a power tool, it works wonders even when the algorithm is not optimally designed but tuning the hyper parameters is just as important as training the model. How do you know which network architecture is the best for the problem at hand? (Think of fitting a function to some data, should you optimize a linear function or a quadratic?) Having too many neurons can lead to overfitting your data, where the network's knowledge goes beyond generalizing trends and starts to learn explicit details (e.g. the shape of the noise) in order to make decisions. Think of a smoothing algorithm (e.g. a spline or savitsky-golay filter) that has too short of a length scale, it captures the details really well but fails to encapsulate a smoother general trend so when you go to extrapolate from this model it could be highly biased. Or if you have too few neurons the algorithm generalizes too much and it's inaccurate. Achieving the proper balance between memorization and generalization will depend on the problem at hand and what you want to do with the algorithm. One way to determine this difference is to use an attention map or [class activation map](https://jacobgil.github.io/deeplearning/class-activation-maps). It basically determines what features from your input space are most relevant for the network to make a decision. 

Use a 2D CNN with phase folded light curves to detect periodic signals (think Period vs. Probability graph) where the input into the CNN would be a phase folded light curve stacked in an image like a 'water-fall' plot. Then the network outputs a probability for when the signal is in phase/aligned (i.e. correct period for phase fold). Training would consist of creating phase folded light curves at correct and incorrect periods. A positive detection would entail having an aligned signal (i.e. correct period) and vice versa for a negative detection. The power of a 2D CNN is that it will learn to correlate features in time and phase. This is particularly important because for a periodic signal, the data will be correlated in phase thus processing the signal should be done with convolutions. Since the data is already correlated in time and will be correlated in phase for the correct period, a 2D CNN is perfect to use.

Using this to detect exoplanet transits and timing transit variations in the next generation of transit survey (e.g. TESS). 
