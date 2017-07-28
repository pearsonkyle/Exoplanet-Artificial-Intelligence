# Searching for Exoplanets using Artificial Intelligence

White Paper Link: (Currently in review at MNRAS) 

arXiv: https://arxiv.org/abs/1706.04319

This project uses machine learning to search for transiting exoplanets in planetary search surveys. Multiple neural networks (MLP, CNN, Wavelet MLP) are trained to recognize patterns from artificial light curves that mimic real observations. The trained networks are validated with real data using the known ephemerii of transiting planets discovered from the [Kepler](https://www.nasa.gov/mission_pages/kepler/main/index.html) mission.

Dependencies: 
  * [Numpy](http://www.numpy.org/)
  * [Keras](https://keras.io/)
  * [TensorFlow](https://www.tensorflow.org/)
  * [Matplotlib](https://matplotlib.org/)
  * [Scikit-learn](http://scikit-learn.org/stable/)
  * [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
  * [Exoplanet Light Curve Analysis](https://github.com/pearsonkyle/Exoplanet-Light-Curve-Analysis)

## Comprehensive File Guide 
`generate_data.py` - generates over 300,000 training and test samples from the parameter space grid in Table 1.

`model_fit_history` - creation of each neural network and its training. The training performance per epoch is saved and used to create Figure 6. 
Files will be released upon publication
