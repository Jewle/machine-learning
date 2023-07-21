import tensorflow as tf
import math

x=tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    x = x + 7
    y = x*x*x
grad = tape.gradient(y,x)
print(grad)    

