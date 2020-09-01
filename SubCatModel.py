import pandas as pd

df = pd.read_csv("styles.csv",error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.sample(frac=1).reset_index(drop=True)
df.head(10)

batch_size = 64

from keras_preprocessing.image import ImageDataGenerator

image_generator = ImageDataGenerator(
    validation_split=0.2
)

training_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory="images",
    x_col="image",
    y_col="subCategory",
    target_size=(60,80),
    batch_size=batch_size,
    subset="training"
)

validation_generator = image_generator.flow_from_dataframe(
    dataframe=df,
    directory="images",
    x_col="image",
    y_col="subCategory",
    target_size=(60,80),
    batch_size=batch_size,
    subset="validation"
)

classes = len(training_generator.class_indices)
from keras.models import Sequential, Model

# network parameters
input_shape = (60, 80, 3)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 256
dropout = 0.2

filters = 32
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D,Dropout,Activation

# model is a 3-layer MLP with ReLU and dropout after each layer
model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(MaxPooling2D(pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(Flatten())
# dropout added as regularizer
model.add(Dropout(dropout))
# output layer is 10-dim one-hot vector
model.add(Dense(classes))
model.add(Activation('softmax'))
model.summary()
# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from math import ceil

model.fit_generator(
    generator=training_generator,
    steps_per_epoch=ceil(0.8 * (df.shape[0] / batch_size)),

    epochs=5,
    verbose=1
)

model.save('mysubnewmodel1.h5')

loss, acc = model.evaluate_generator(validation_generator, steps=ceil(0.2 * (df.size / batch_size)))
print("\n%s: %.2f%%" % (model.metrics_names[1], acc * 100))

import numpy as np
from keras.preprocessing import image
filename='55133.jpg'
test_image = image.load_img('images\\'+filename,target_size=(60,80))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(np.argmax(result))
print(training_generator.class_indices)


my_dict = training_generator.class_indices
key_list = list(my_dict.keys()) 
val_list = list(my_dict.values()) 

import pickle
# save the model to disk
filename1 = 'skey_list'
filename = 'sval_list'
pickle.dump(key_list, open(filename1, 'wb'))
pickle.dump(val_list, open(filename, 'wb'))
