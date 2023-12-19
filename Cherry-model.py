import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

image_size = 224
batch_size = 16
CHANNELS = 3
EPOCHS = 5

train_data = "Data/train"
val_data = "Data/valid"
train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        shear_range= 0.2,
        horizontal_flip = True
)
test_val_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    directory = train_data,
    target_size=(image_size, image_size),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True
)
val_generator = test_val_datagen.flow_from_directory(
    directory = val_data,
    target_size=(image_size, image_size),
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False
)
print("******************************************************************************************")
print("")
print("Getting the model")
print("******************************************************************************************")
print("")
net = keras.applications.mobilenet_v2.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(image_size, image_size, 3)
)
net.summary()
for layer in net.layers[:-2]:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(net.output)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
prediction = layers.Dense(2, activation="softmax")(x)
model = models.Model(inputs = net.input, outputs = prediction)
model.summary()

model.compile(
    loss = "categorical_crossentropy",
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics = ['acc']
)
print("******************************************************************************************")
print("")
print("Training")
print("******************************************************************************************")
print("")

history = model.fit(
    train_generator,
    epochs = 5,
    validation_data=val_generator
)
print("******************************************************************************************")
print("")
print("Evaluating")
print("******************************************************************************************")
print("")
scores = model.evaluate(val_generator, verbose = 1)
print(f"Loss : {scores[0]} Accuracy:{scores[1]}")
print("******************************************************************************************")
print("")
print("Saving Model")
print("******************************************************************************************")
print("")
models.save_model(model, "model/CherryLeafMobileNet.h5")
print("Model Saved..")
print("******************************************************************************************")
print("Testing")
print("******************************************************************************************")
print("")
test_pred = model.predict(val_generator, verbose = 1)
test_labels = np.argmax(test_pred, axis=1)
print("******************************************************************************************")
print("")
print("Prediction Report")
print("******************************************************************************************")
print("")
class_labels = val_generator.class_indices
class_labels = {v:k for k,v in class_labels.items()}
classes = list(class_labels.values())
print(classes)
print('Classification Report')
print(classification_report(val_generator.classes, test_labels, target_names=classes))
















