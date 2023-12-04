# --------------------------------------------------------------
# import required libraries
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------------------
# load model
# --------------------------------------------------------------

model_pkl_file = "../../models/food_classifier_model.pkl"
model = pickle.load(open(model_pkl_file, 'rb'))

# --------------------------------------------------------------
# load test dataframe
# --------------------------------------------------------------

test_df = pd.read_csv("../../data/processed/test_data.csv")

# --------------------------------------------------------------
# generating batches of images for testing the model
# --------------------------------------------------------------

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# --------------------------------------------------------------
# Loading Testing Data
# --------------------------------------------------------------

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepaths',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# --------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------

results = model.evaluate(test_images, verbose=0)

# --------------------------------------------------------------
# Predictions, Confusion Matrix and Classification Report
# --------------------------------------------------------------

predictions = np.argmax(model.predict(test_images), axis=1)

cm = confusion_matrix(test_images.labels, predictions)
clr = classification_report(test_images.labels, predictions,
                            target_names=test_images.class_indices, zero_division=0)

# --------------------------------------------------------------
# plot confusion matrix
# --------------------------------------------------------------

plt.figure(figsize=(30, 30))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(20) + 0.5,
           labels=test_images.class_indices, rotation=90)
plt.yticks(ticks=np.arange(20) + 0.5,
           labels=test_images.class_indices, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(
    f"../../reports/figures/Confusion matrix.png", bbox_inches='tight')
plt.show()


# --------------------------------------------------------------
# classification report
# --------------------------------------------------------------

print("Classification Report:\n----------------------\n", clr)
