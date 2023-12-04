# --------------------------------------------------------------
# import required libraries
# --------------------------------------------------------------

import pandas as pd
from pathlib import Path
import os.path

# --------------------------------------------------------------
# create dataframe
# --------------------------------------------------------------

image_dir = Path('../../data/raw/FoodClassification/')

filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepaths').astype(str)
labels = pd.Series(labels, name='Label')
images = pd.concat([filepaths, labels], axis=1)
category_samples = []

for category in images['Label'].unique():
    category_slices = images.query('Label==@category')
    category_samples.append(category_slices.sample(frac=1, random_state=1))
image_df = pd.concat(category_samples,axis=0).sample(frac=1.0,random_state=1).reset_index(drop=True)


# --------------------------------------------------------------
# export the dataframe
# --------------------------------------------------------------

image_df.to_csv('../../data/interim/data.csv')
