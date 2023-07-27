from tensorflow import keras
from keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def get_mnist_model():#определение начальных параметров модели
 inputs = keras.Input(shape=(28 * 28,))#форма входных данных (784,)
 features = layers.Dense(512, activation="relu")(inputs)#передаем входные данный в 512 мерный слой
 features = layers.Dropout(0.5)(features)#прореживание
 outputs = layers.Dense(10, activation="softmax")(features)#определяем выходные данные
 model = keras.Model(inputs, outputs)#инициализируем модель
 return model

(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()

#Набор встроенных колбеков
callbacks_list = [
 keras.callbacks.EarlyStopping(
 monitor="val_accuracy",
 patience=2,
 ),#Автоматиеская остановка  призвана избежать переобучения
]


loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()#среднее значение ошибки


def train_step(inputs, targets):
 with tf.GradientTape() as tape: #Для вычисления градиента необходим конекст
     predictions = model(inputs, training=True)# Важно указать true для training
     loss = loss_fn(targets, predictions)
 gradients = tape.gradient(loss, model.trainable_weights)#Нахождение градиентов
 optimizer.apply_gradients(zip(gradients, model.trainable_weights))#Применение градиентов с учетом оптимизатора (так написано в документации)
 logs = {}
 for metric in metrics:
     metric.update_state(targets, predictions)
     logs[metric.name] = metric.result()
     loss_tracking_metric.update_state(loss)
     logs["loss"] = loss_tracking_metric.result()
 return logs

def reset_metrics():
 for metric in metrics:
     metric.reset_state()
     loss_tracking_metric.reset_state()

training_dataset = tf.data.Dataset.from_tensor_slices(
 (train_images, train_labels))

training_dataset = training_dataset.batch(32)#разделяет массив на 32 подмассива (бача)
epochs = 3

#Цикл обучения 
for epoch in range(epochs):
  reset_metrics()
  for inputs_batch, targets_batch in training_dataset:
      logs = train_step(inputs_batch, targets_batch)
      print(f"Results at the end of epoch {epoch}")
      for key, value in logs.items():
          print(f"...{key}: {value:.4f}"

def test_step(inputs, targets):
    predictions = model(inputs, training=False)
   loss = loss_fn(targets, predictions)
   logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["val_" + metric.name] = metric.result()
        oss_tracking_metric.update_state(loss)
        logs["val_loss"] = loss_tracking_metric.result()
    return logs

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()
for inputs_batch, targets_batch in val_dataset:
    logs = test_step(inputs_batch, targets_batch)
    print("Evaluation results:")
for key, value in logs.items():
    print(f"...{key}: {value:.4f}")                



# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, "r", label="Потери на этапе обучения")
# plt.plot(epochs, val_loss_values, "b", label="Потери на этапе проверки")
# plt.title("Потери на этапах обучения и проверки")
# plt.xlabel("Эпохи")
# plt.ylabel("Потери")
# plt.legend()
# plt.show()






# test_metrics = model.evaluate(test_images, test_labels)
# predictions = model.predict(test_images)



