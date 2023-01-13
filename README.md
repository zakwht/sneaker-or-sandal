# Sneaker or Sandal SVM Classifier

[![License](https://img.shields.io/github/license/zakwht/sneaker-or-sandal)](/LICENSE.md)
![Accuracy](https://img.shields.io/badge/accuracy-98.4%25-mediumgreen)

A classification model that uses a machine learning to categorize an image of a shoe as either a sneaker or a sandal. The app converts images into a 2D array of pixels (28x28) which are classified by a support vector machine (SVM) powered by a Gaussian (RBF) kernel. 
* The model's parameters are the regularization parameter `C = 4` and the kernel coefficient `γ = 0.036`. These were the results of hyperparameter tuning with seven-fold cross-validation.
* The training set had 4000 images, and the test set had 667 (6:1 split). The model ultimately performed with 98.4% accuracy based on the data.
* The 95% confidence interval for the test error is `(0, 0.0687)`.


![Demo](/static/sample.png)

### Development

Requirements are cataloged in [requirements.txt](./requirements.txt).

The classifier can be run from the command line with `make`, which will run the site from _localhost:5000_.

### Acknowledgments
* __Model trained using__ the [Fashion-NMIST](https://github.com/zalandoresearch/fashion-mnist) dataset.
* __Implemented with__ the [scikit-learn](https://scikit-learn.org/stable/) machine learning package for Python