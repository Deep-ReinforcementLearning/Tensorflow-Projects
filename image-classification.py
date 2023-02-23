import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# importing clothing dataset from servers reason:we are doing image classification
data = keras.datasets.fashion_mnist

# spliting data to train and test reason: no memorising
(train_images, train_labels), (test_images, test_labels) = data.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print the first label reason: test if the data is loaded
#code: print(train_labels[0])

train_images = train_images/255.0
train_images = test_images/255.0

#print(train_images[7])
#show the images by matplot lib reason: test if the data is loaded
#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()
#defining the layers of the model input hidden activation output flatten the data
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    #dense layer & activation layer which we will use is rectified linear unit 
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
#setup the optimizer and loss function
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]


#train the model 
# adding epochs
fitModel = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5")


#test acc= accuracy
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested accuracy :", test_acc)

#prediction pass a list



#output



'''
to study
activation function softmax
optimizer adam


'''
