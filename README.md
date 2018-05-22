# CNN 

Explore using Convolutional Neural Networks to optimize feature selection jointly with learning a
classification policy.
Denote the input state x as 90x90x3 RGB image, which is a down sampled RGB image with the fruit centered
in it. Each data point will have a corresponding class label, which corresponds to their matching
produce. Given 25 classes, we can denote the label as y in { 0,...24}
The goal of this problem is twofold. First we will learn how to implement a Convolutional Neural
Network (CNN) using TensorFlow. Then we will explore some of the mysteries about why neural
networks work as well as they do in the context of a bias variance trade-off.


(A) To begin the problem, we need to implement a CNN in TensorFlow. In order to reduce the
burden of implementation, we are going to use a TensorFlow wrapper known as slim. In
the starter code is a file named cnn.py, the network architecture and the loss function are
currently blank. Using the slim library, you will have to write a convolutional neural network
that has the following architecture:
    (a) Layer 1: A convolutional layer with 5 filters of size 15 by 15
    (b) Non-Linear Response: Rectified Linear Units
    (c) A max pooling operation with filter size of 3 by 3
    (d) Layer 2: A Fully Connected Layer with output size 512.
    (e) Non-Linear Response: Rectified Linear Units
    (f) Layer 3: A Fully Connected Layer with output size 25 (i.e. the class labels)
    (g) Loss Layer: Softmax Cross Entropy Loss
In the file example cnn.py, we show how to implement a network in TensorFlow Slim. Please
use this as a reference. Once the network is implemented run the script test_cnn.py
on the dataset and report the resulting confusion matrix. The goal is to ensure that your
network compiles, but we should not expect the results to be good because it is randomly
initialized.
(B) The next step to train the network is to complete the pipeline which loads the datasets and
offers it as mini-batches into the network. Fill in the missing code in data_manager.py and
report your code.

(C) We will now complete the iterative optimization loop. Fill in the missing code in trainer.py
to iteratively apply SGD for a fix number of iterations. In our system, we will be using an
extra Momentum term to help speed up the SGD optimization. Run the file train_cnn.py
and report the resulting chart.

(D) To better understand, how the network was able to achieve the best performance on our
fruits and veggies dataset. It is important to understand that it is learning features to reduce
the dimensionality of the data. We can see what features were learned by examining the
response maps after our convolutional layer.

  The response map is the output image after the convolutional has been applied. This image
can be interpreted as what features are interesting for classification. Fill in the missing code
in visualized_features.py and report the images specified.

(e) Given that our network has achieved high generalization with such low training error, it
suggests that a high variance estimator is appropriate for the task. To better understand why
the network is able to work, we can compare it to another high variance estimator such as
nearest neighbors. Fill in the missing code in nn_classifier.py and report the performance
as the numbers of neighbors is swept across when train_nn.py is run.
