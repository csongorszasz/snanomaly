### Data Loading and Preprocessing

- understand how cattrs structure hooks work
- fix parsing issues that are fixable
- validate objects for adherence to requirements before passing them on
  - has photometric data 
  - has observations with required filters
    - B, V, R, I
    - etc. # TODO
  - minimum 3 photometric observations per band
  - etc. # TODO

- create unit tests for the models
- create unit tests for parsing complex objects (.json)

- restructure models to reduce number of uninitialized attributes
- employ a generator for loading and processing of data instead of loading all objects into memory
  - at larger datasets loading all objects into memory is not feasible

### Photometry Lightcurve Reconstruction - Regression
- lightcurve reconstruction with:
  - Gaussian Processes
  - Kernel Ridge Regression
  - Support Vector Machines
  - Gradient Boosing Regression
- visualization of reconstructed lightcurves

### Dimensionality Reduction
- dimensionality reduction with:
  - t-SNE
  - PCA
  - UMAP
  - LLE
- visualization of reduced data

### Outlier detection
- outlier detection with: 
  - Gaussian Processes
  - Isolation Forest
  - One-Class SVM
  - Neural Networks
- visualization of outliers
