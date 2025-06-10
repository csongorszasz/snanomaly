### Data Loading and Preprocessing

- ~~understand how cattrs structure hooks work~~
- ~~setup logging correctly~~ 
- ~~fix parsing issues that are fixable~~
- ~~validate objects for adherence to requirements before passing them on~~
  - ~~has photometric data~~ 
  - ~~minimum 3 photometric observations per band with 3-day binning~~

- ~~binned curve plotting~~
  - ~~plot binned dots~~ 
  - ~~get LaTeX to work in plots~~

- ~~band transformation~~

- create unit tests for the models
- create unit tests for parsing complex objects (.json)

- restructure models to reduce number of uninitialized attributes
- ~~employ a generator for loading and processing of data instead of loading all objects into memory~~

### Photometry Lightcurve Reconstruction - Regression
- lightcurve reconstruction with:
  - ~~Gaussian Processes~~
  - ~~Kernel Ridge Regression~~
  - Support Vector Machines
  - ~~Gradient Boosting Regression~~
- ~~visualization of reconstructed lightcurves~~
  - ~~error bars for ground truth data points~~

### Dimensionality Reduction
- dimensionality reduction with:
  - ~~t-SNE~~
  - PCA
  - UMAP
  - Isomap
  - LLE
- visualization of reduced data

### Outlier detection
- outlier detection with:
  - ~~Isolation Forest~~
  - One-Class SVM
  - Neural Networks
- visualization of outliers


### Possibilities for improvement
1. ~~! Handle upper limits differently from normal observations~~
2. ~~! Cut down fitting interval or do something so that distant observations don't affect the predictions~~
3. ~~! Add ignored upper limits colored in black~~
4. ~~! Add black outline to interpolation plots~~
5. ~~! Modify the legend in the interpolation plots so that it doesn't overlap the light curves~~
6. ~~! Fix formatting of y axes text in light curve plot~~ 
7. ! Experiment with ARD for GPR hyperparameter setting
8. ! Experiment with Grid Search and similar methods for KRR for hyperparameter setting
9. Calibrate magnitude using zero-point magnitudes (m_calibrated = m_instrumental - m_zeropoint)
   - difficult because not every observation comes with zero-point information
10. Convert count rate to flux
11. Convert flux density to flux
12. Involve spectral data in the outlier detection
