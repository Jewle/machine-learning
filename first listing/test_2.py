import numpy as np
import tensorflow as tf
from NaiveDense import *
from NaiveSequential import *
import math

print(NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax))


        


        
model = NaiveSequential([
 NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
 NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4

print(model.__call__(np.array([6,3,5,1])))

