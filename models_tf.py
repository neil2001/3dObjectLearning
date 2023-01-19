

import tensorflow as tf
import numpy as np
import random
import math

class Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, dims, l2Val=0.001):
        self.dim = dims
        self.regVal = l2Val
        self.idMat = tf.eye(self.dim)

    def __call__(self, x): 
        A = tf.reshape(x, (-1, self.dim, self.dim))
        # print(A, A.shape)
        AAt = tf.tensordot(A, A, axes=(2, 2))
        AAt = tf.reshape(AAt, (-1, self.dim, self.dim))
        return tf.reduce_sum(self.regVal * tf.square(AAt - self.idMat))


class TNet_TF(tf.keras.Model):
    def __init__(self, dims=3, multiplier=1.0):
        super().__init__()
        self.dim = int(dims)

        bias = tf.keras.initializers.Constant(np.eye(self.dim).flatten())
        reg = Regularizer(self.dim)

        self.conv1 = tf.keras.layers.Conv1D(32 * multiplier, kernel_size=1, padding="valid")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu1 = tf.keras.layers.Activation("relu")

        self.conv2 = tf.keras.layers.Conv1D(64 * multiplier, kernel_size=1, padding="valid")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu2 = tf.keras.layers.Activation("relu")

        self.conv3 = tf.keras.layers.Conv1D(512 * multiplier,1)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu3 = tf.keras.layers.Activation("relu")

        self.maxpool4 = tf.keras.layers.GlobalMaxPooling1D()

        self.linear5 = tf.keras.layers.Dense(256 * multiplier)
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu5 = tf.keras.layers.Activation("relu")

        self.linear6 = tf.keras.layers.Dense(128 * multiplier)
        self.bn6 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu6 = tf.keras.layers.Activation("relu")

        self.linear7 = tf.keras.layers.Dense(
                            self.dim * self.dim,
                            kernel_initializer="zeros",
                            bias_initializer=bias,
                            activity_regularizer=reg,
                        )

        self.reshape = tf.keras.layers.Reshape((self.dim, self.dim))
        self.dotProd = tf.keras.layers.Dot(axes=(2,1))

    def call(self, inputs, training=False):
        x = self.relu1(self.bn1(self.conv1(inputs), training=training))
        x = self.relu2(self.bn2(self.conv2(x), training=training))
        x = self.relu3(self.bn3(self.conv3(x), training=training))

        x = self.maxpool4(x)
        x = self.relu5(self.bn5(self.linear5(x), training=training))
        x = self.relu6(self.bn6(self.linear6(x), training=training))

        x = self.linear7(x)

        features = self.reshape(x)
        return self.dotProd([inputs, features])

class PointNet_TF(tf.keras.Model):
    def __init__(self, num_classes, multiplier = 1.0, tnet_mult = 1.0):
        super().__init__()

        self.batch_size = 32
        self.num_classes = num_classes
        
        self.tnet3 = TNet_TF(3, multiplier=tnet_mult)

        self.conv1 = tf.keras.layers.Conv1D(32 * multiplier,1)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv1D(32 * multiplier,1)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu2 = tf.keras.layers.ReLU()

        self.tnet64 = TNet_TF(32 * multiplier, tnet_mult)

        self.conv3 = tf.keras.layers.Conv1D(32 * multiplier, 1)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu3 = tf.keras.layers.ReLU()

        self.conv4 = tf.keras.layers.Conv1D(64 * multiplier, 1)
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu4 = tf.keras.layers.ReLU()

        self.conv5 = tf.keras.layers.Conv1D(512 * multiplier,1)
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu5 = tf.keras.layers.ReLU()

        self.maxPool6 = tf.keras.layers.GlobalMaxPooling1D()
            
        self.linear7 = tf.keras.layers.Dense(256 * multiplier)
        self.bn7 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu7 = tf.keras.layers.ReLU()

        self.linear8 = tf.keras.layers.Dense(128 * multiplier)
        self.bn8 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.relu8 = tf.keras.layers.ReLU()

        self.dropout9 = tf.keras.layers.Dropout(0.3)

        self.softmax10 = tf.keras.layers.Dense(num_classes, activation="softmax")
        pass

    def call(self, x, training=False):
        x = self.tnet3(x) #tf.convert_to_tensor(self.tnet3(x), dtype = tf.float32)
        x = self.relu1(self.bn1(self.conv1(x), training=training))
        x = self.relu2(self.bn2(self.conv2(x), training=training))
        x = self.tnet64(x)
        x = self.relu3(self.bn3(self.conv3(x), training=training))
        x = self.relu4(self.bn4(self.conv4(x), training=training))
        x = self.relu5(self.bn5(self.conv5(x), training=training))

        x = self.maxPool6(x)

        x = self.relu7(self.bn7(self.linear7(x), training=training))
        x = self.relu8(self.bn8(self.linear8(x), training=training))

        x = self.dropout9(x, training=training)

        x = self.softmax10(x)
        return x

    def accuracy(self, logits, labels):
        print("IN ACCURACY, NUM CLASSES", self.num_classes)
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
