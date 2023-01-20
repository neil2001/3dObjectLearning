# 3dObjectLearning
For the full writeup, visit this link: https://docs.google.com/document/d/1UVj50E6Kaw9PEmVhHH8ppt3W_tMlm4vP-7bubc3wrSo/edit?usp=sharing

# Usage:
You may need to create the following directories:

`data`, `images/confusion_shapes`, `images/confusion_snow`, `shapeModel`, `snowModel`

To train and save the model using the snow dataset, run:

`python3 main_tf.py snow train`

To observe the classification results on testing data, run:

`python3 main_tf.py snow`

To train the model and observe behavior on the shapes dataset, run the commands but with

`shapes`

as the second argument

# Epochs vs. Accuracy on Tensorflow Model with 3 simple objects (cone, sphere, cube)
1 epoch: 87% training accuracy, 68% testing accuracy

2 epochs: 95% training accuracy, 87% testing accuracy

3 epochs: 98% training accuracy, 97% testing accuracy

4 epochs: 99% training accuracy, 99% testing accuracy

# Epochs vs. Accuracy on Tensorflow Model with 4 Simple Objects (cone, sphere, cube, cylinder)

1 epoch: 82% training accuracy, 94% testing accuracy

2 epochs: 95% training accuracy,98% testing accuracy

3 epochs: 98% training accuracy, 98% testing accuracy

4 epochs: 99% training accuracy, 97% testing accuracy

# Epochs vs. Accuracy on Tensorflow Model with 3 complex objects (cloud, snowman, ice cream cone)

1 epoch: 84% training accuracy, 84% testing accuracy

2 epochs: 92% training accuracy, 80% testing accuracy

3 epochs: 94% training accuracy, 96% testing accuracy

4 epochs: 97% training accuracy, 96% testing accuracy

5 epochs: 97% training accuracy, 99% testing accuracy

6 epochs: 99% training accuracy, 99% testing accuracy


6 epochs is not always consistent,
with 8 epochs, was able to achieve 99% accuracy --> with these only the sideways snowmen
were predicted as clouds. No other confusion happened.
