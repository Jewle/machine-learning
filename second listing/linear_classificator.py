import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


num_samples_per_class = 2#кол-во точек
negative_samples = np.random.multivariate_normal(
 mean=[0, 3],
 cov=[[1, 0.5],[0.5, 1]],
 size=num_samples_per_class) #тысяча точек первого класса
positive_samples = np.random.multivariate_normal(
 mean=[3, 0],
 cov=[[1, 0.5],[0.5, 1]],
 size=num_samples_per_class) # тысяча точек второго класса

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
 np.ones((num_samples_per_class, 1), dtype="float32")))

# print(inputs)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets)
# plt.show()



input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))



def model(inputs):#выход вектор (num_samples_per_class*2,1)
  return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):# выход - скаляр, средняя общая ошибка для одной точки
  per_sample_losses = tf.square(targets - predictions)
  return tf.reduce_mean(per_sample_losses)





learning_rate = 0.1
def training_step(inputs, targets):
  with tf.GradientTape() as tape:
      predictions = model(inputs)
      loss = square_loss(predictions, targets)
  grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
  print(grad_loss_wrt_W)
  W.assign_sub(grad_loss_wrt_W * learning_rate)
  b.assign_sub(grad_loss_wrt_b * learning_rate)
  return loss

training_step(inputs, targets)
# def training_steps(times):
#     for step in range(times):
#          loss = training_step(inputs, targets)
#          # print(loss)
         
         
# training_steps(100)

# print(W)
# x = np.linspace(-1, 4, 100)
# y = - W[0] / W[1] * x + (0.5 - b) / W[1]
# plt.plot(x, y, "-r")
# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets)
# plt.show()


