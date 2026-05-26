# Lecture 2 -- Image Classification

## Intro
**Goal**: take in an image as the input and outputs a classifier(e.g. 'cat', 'saki')
```
def classify_image(image):
    # your code here......
    return class_label
```

**Main challenge**: 
1. semantic gap; what the computer sees is a big grid of numbers between [0,255] (e.g. 800 x 600 x 3: 3 channels RGB)
2. inclass variation: 
3. fine-grained categories: classify between similar classes, like different breeds of cats
4. background clutter: objects may mixed up with the environment
5. illumination changes: caused by lights
6. deformation: objects may be deformable, in different poses
7. ......

**Value**: a useful building block that can be applied in many other algorithms

## Method: Data-Driven Approach

**Pipeline**:
1. Collect a dataset of images and labels
2. Use Machine Learning to **train** a classifier
3. Evaluate the classifier on **new images**

Instead of designing a single "classify_image" function as above, we would try:
```
def train(images, labels):
    # Machine Learning...
    return model

def predict(model, test_images):
    # Use model to predict labels...
    return test_labels
```

The follow are some famous image classification datasets you might want to train/test your algorithms on:
1. MNIST: 10 classes, 28x28 grayscale images, 50k training images, 10k test images
2. CIFAR10: 10 classes(airplane, automobile, bird, cat, ...), 32x32 RGB images, 50k training images, 10k test images(**Note: will be used throughout sll the projects of this course**)
3. CIFAR100: 100 classes, ...
4. ImageNet: 1000 classes, $\approx$1.3M training images, 50k validation images, 100k test images; performance metric: top 5 accuracy
5. MIT places: 365 classes of different **scene** types
6. Omniglot: 1623 categories while only 20 images per catrgory! meant to test **few shot learning**

### First classifier: Nearest Neighbour

1. **How it works**: In train(...) function, we would memorize all training data and labels, and the predict(...) function would select the label of the most similar training image(or several of the most, and then predict by voting) as the result 
2. **Methods needed**: distance matrics to compare images($L_p$ distances, e.t.c.)
3. **Implementation**:
```{python}
import numpy as np

def l1_distance(X_train, x_test_single):
    return np.sum(np.abs(X_train - x_test_single), axis=1)

def l2_distance(X_train, x_test_single):
    return np.sqrt(np.sum(np.square(X_train - x_test_single), axis=1))

class NearestNeighbour:
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-D of size N."""
        self.Xtr = X
        self.ytr = y

    def compute_distance(self, X_train, x_test_single, dist_func):
        return dist_func(X_train, x_test_single)

    def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for. """
    num_test = X.shape[0]
    # make sure that the output tupe matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.type)

    # loop over all test rows
    for i in range(num_test):
        # find the nearest training image to the i^th test image
        distances = self.compute_distance(self.Xtr, X[i, :], dist_func)
        min_index = np.argmin(distances) # get the index with smallest distance
        Ypred[i] = self.ytr[min_index]

    return Ypred
```
Notice that with N examples, the time complexity for training is O(1) (if use shallow copy) while for testing it's O(N). This is **BAD** since we can afford slow training, but we always need **fast testing**.

4. **Ways to improve/change**: change the number of neighbours(k), change the metric you use(d(\*,\*), e.g. if you want to calculate the distance between pdfs/texts, you may use *TF-IDF similarity*), ...

These are examples of **hyperparameters**(unfortunately they are very problem-dependent). To get the a better hyperparameter, we may split data into *training* and *validation* sets, and choose hyperpatameters that work best on validation sets.(Note that test data could not be used in validation set, for in this case the model would 'know' the test data)

A even better method is to use cross-validation: Split data into **folds**, try each fold as validation and average the result. (Note: you have to select a portion of the data to be fixed as test sets that won't be used in the tuning&training process) This is useful for small datasets, but extremely time consuming when working on complex problem.

5. Universal Approximation: As the number of training samples goes to infinity, nearest neighbor can represent lots of functions correctly. However, there's a curse of dimensionality: for uniform convergence, number of training points needed grows exponentially with dimention(since you have to 'cover' the points in the domain)

6. **Why seldom used**: 
    * very slow at test time
    * distance metrics on raw pixels are not imformative
    
However, Nearest Neighbours with **ConvNet features**(feature vectors computed from deep neural networks) works well! 