# from tensorflow.keras.datasets import mnist
from numpy import *
from matplotlib import pyplot as plt
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# plt.imshow(train_images[4], interpolation='nearest')
# plt.show()

grad = gradient(array([[1,2,3],[4,5,6], [7,8,9]]))
print(grad)
plt.imshow(grad, interpolation='nearest')
plt.show()
array([1,2])