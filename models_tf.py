

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
    def __init__(self, dims=3):
        super(TNet_TF, self).__init__()
        self.dim = dims

        bias = tf.keras.initializers.Constant(np.eye(self.dim).flatten())
        reg = Regularizer(self.dim)

        self.mlp1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(64, kernel_size=1, padding="valid"),
                tf.keras.layers.BatchNormalization(momentum=0.0),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.Conv1D(128, kernel_size=1, padding="valid"),
                tf.keras.layers.BatchNormalization(momentum=0.0),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.Conv1D(1024,1),
                tf.keras.layers.BatchNormalization(momentum=0.0),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.GlobalMaxPooling1D(),

                tf.keras.layers.Dense(512),
                tf.keras.layers.BatchNormalization(momentum=0.0),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.Dense(256),
                tf.keras.layers.BatchNormalization(momentum=0.0),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.Dense(
                    self.dim * self.dim,
                    kernel_initializer="zeros",
                    bias_initializer=bias,
                    activity_regularizer=reg,
                )
            ]
        )

        self.reshape = tf.keras.layers.Reshape((self.dim, self.dim))
        self.dotProd = tf.keras.layers.Dot(axes=(2,1))

    def call(self, inputs):

        x = self.mlp1(inputs)
        features = self.reshape(x)
        return self.dotProd([inputs, features])

class PointNet_TF(tf.keras.Model):
    def __init__(self, num_classes):
        super(PointNet_TF, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.input_size = 1024
        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.tnet3 = TNet_TF()
        self.tnet64 = TNet_TF(64)

        self.mlp1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(64,1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv1D(64,1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )

        self.mlp2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(64, 1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv1D(128, 1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv1D(1024,1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.GlobalMaxPooling1D(),
            
                tf.keras.layers.Dense(512),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Dense(256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Dense(num_classes, activation="softmax")
            ]
        )
        pass

    def call(self, x):
        transform = self.tnet3(x) #tf.convert_to_tensor(self.tnet3(x), dtype = tf.float32)
        x = self.mlp1(x)
        x = self.tnet64(x)
        x = self.mlp2(x)
        return x

    def loss(self, logits, labels):
        softMaxLogits = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        softMaxLogits = tf.reduce_mean(softMaxLogits)
        # print(softMaxLogits)
        return softMaxLogits

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels, num_epochs):
    # losses = []
    # accuracy_vals = []
    # model.fit()

    for n in range(num_epochs):
        indices = tf.range(0, len(train_inputs), 1)
        shuffledIndices = tf.random.shuffle(indices)
        shuffledInputs = tf.gather(train_inputs, shuffledIndices)
        shuffledLabels = tf.gather(train_labels, shuffledIndices)
        # randomizedInputs =  tf.image.random_flip_left_right(shuffledInputs)
        
        epoch_loss_sum = 0
        epoch_correct_num = 0
        for i in range(0, len(train_inputs), model.batch_size):
            batchInputs = shuffledInputs[i:i+model.batch_size]
            batchLabels = shuffledLabels[i:i+model.batch_size]

            with tf.GradientTape() as tape:
                probs = model.call(batchInputs)
                loss = model.loss(probs, batchLabels)
                # print(loss)
                model.loss_list.append(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
    return

def test(model, test_inputs, test_labels):
    # model.evaluate()
    probs = model.call(test_inputs)
    return model.accuracy(probs, test_labels)