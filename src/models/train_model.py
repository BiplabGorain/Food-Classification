# --------------------------------------------------------------
# import required libraries
# --------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

# --------------------------------------------------------------
# import the dataframe
# --------------------------------------------------------------

image_df = pd.read_csv("../../data/interim/data.csv")

# --------------------------------------------------------------
# Train test split
# --------------------------------------------------------------

train_df,test_df = train_test_split(image_df,train_size=0.7,shuffle=True,random_state=1)

# --------------------------------------------------------------
# generating batches of images for training the model
# --------------------------------------------------------------

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)


# --------------------------------------------------------------
# Loading Training and Validation Data
# --------------------------------------------------------------

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepaths',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepaths',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

# --------------------------------------------------------------
# Load Pre-trained Model
# --------------------------------------------------------------

pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'

)
pretrained_model.trainable = False

# --------------------------------------------------------------
# define layers
# --------------------------------------------------------------

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(20,activation='softmax')(x)

# --------------------------------------------------------------
# creating model
# --------------------------------------------------------------

model = tf.keras.Model(inputs, outputs)
model.summary()

# --------------------------------------------------------------
# train the model
# --------------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

model_pkl_file = "../../models/food_classifier_model.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(model, file)
