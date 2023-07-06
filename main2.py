from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from keras import models
from keras import layers

from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))),

# first_layer_model = models.Model(
#     inputs=model.input, outputs=model.layers[0].output)

# # Assuming you have an input image 'input_image'
# # ...

# # Load and preprocess the image
# image = Image.open('33.jpg')
# image = image.resize((28, 28))
# image = image.convert('L')  # Convert to grayscale
# image = np.array(image) / 255.0  # Normalize pixel values

# # Prepare the image for prediction
# image = image.reshape(1, 28, 28, 1)  # Reshape and add batch dimension


# # Pass the input image through the first layer
# first_layer_output = first_layer_model.predict(image)

# # Reshape the output to have a shape (height, width, num_filters)
# first_layer_output = np.squeeze(first_layer_output)
# height, width, num_filters = first_layer_output.shape

# # Plotting the output of each filter
# plt.figure(figsize=(10, 6))
# for i in range(num_filters):
#     plt.subplot(4, 8, i + 1)  # Assuming 32 filters, adjust as needed
#     plt.imshow(first_layer_output[:, :, i], cmap='gray')
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

model.add(layers.MaxPooling2D((2, 2))),
model.add(layers.Conv2D(64, (3, 3), activation='relu')),
model.add(layers.MaxPooling2D((2, 2))),
model.add(layers.Conv2D(64, (3, 3), activation='relu')),
model.add(layers.Flatten()),
model.add(layers.Dense(64, activation='relu')),
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

model.evaluate(x_test, y_test)
