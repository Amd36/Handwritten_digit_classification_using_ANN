import numpy as np
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
import tensorflow as tf

# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


# Set file paths based on added MNIST Datasets
input_path = r"F:\Surveillence_System_Project_(BCSIR)\test_model1\mnist_digit_recognition"
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_test_mod = np.array(x_test)
x_train_mod = np.array(x_train)

y_test_mod = np.array(y_test)
y_train_mod = np.array(y_train)

x_test_mod = np.reshape(x_test_mod, (x_test_mod.shape[0], x_test_mod.shape[1]*x_test_mod.shape[2]))
x_train_mod = np.reshape(x_train_mod, (x_train_mod.shape[0], x_train_mod.shape[1]*x_train_mod.shape[2]))

# Normalize the pixel values
x_train_mod = x_train_mod/255
x_test_mod = x_test_mod/255

# Hot-encode the y values
y_train_mod = np.zeros((y_train_mod.shape[0], 10))

for index in range(0, y_train_mod.shape[0]):
    hot_encode_vect = np.zeros(10)

    hot_encode_vect[y_train[index]] = 1

    y_train_mod[index] = hot_encode_vect

y_test_mod = np.zeros((y_test_mod.shape[0], 10))    

for index in range(0, y_test_mod.shape[0]):
    hot_encode_vect = np.zeros(10)

    hot_encode_vect[y_test[index]] = 1

    y_test_mod[index] = hot_encode_vect


batch_size = 512
epochs = 25
learning_rate = 0.15

# Creating the model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(784, )))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
              loss= tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Training the model
history = model.fit(x=x_train_mod, 
                    y=y_train_mod, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=0.2)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(loss, label="Loss")
plt.title("Loss Reduction")

plt.subplot(2, 1, 2)
plt.plot(val_acc, label='val_accuracy')
plt.title("Value Accuracy")

plt.show()

# Save the model in the current directory
model.save("model.keras")

# Load the same model
loaded_model = tf.keras.models.load_model("model.keras")

# Make Prediction
y_prob = loaded_model.predict(x_test_mod)

loss, accuracy = loaded_model.evaluate(x=x_test_mod, y=y_test_mod)

print(f"The accuracy of the model is {round(accuracy*100, 2)}%")