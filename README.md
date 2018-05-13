# CNN 

Explore using Convolutional Neural Networks to optimize feature selection jointly with learning a
classification policy.
Denote the input state x in R^90*90*3, which is a down sampled RGB image with the fruit centered
in it. Each data point will have a corresponding class label, which corresponds to their matching
produce. Given 25 classes, we can denote the label as y in { 0,...24}
The goal of this problem is twofold. First you will learn how to implement a Convolutional Neural
Network (CNN) using TensorFlow. Then we will explore some of the mysteries about why neural
networks work as well as they do in the context of a bias variance trade-off.
