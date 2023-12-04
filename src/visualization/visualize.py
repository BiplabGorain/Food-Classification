# --------------------------------------------------------------
# import required libraries
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# import the dataframe
# --------------------------------------------------------------

image_df = pd.read_csv("../../data/interim/data.csv")

# --------------------------------------------------------------
# Get the number of images for each label
# --------------------------------------------------------------

label_counts = image_df['Label'].value_counts()

# --------------------------------------------------------------
# Create a bar chart of the label counts
# --------------------------------------------------------------

plt.bar(label_counts.index, label_counts.values)
plt.xlabel('Label')
plt.ylabel('Number of Images')
plt.title('Number of Images for Each Label')
plt.xticks(rotation='vertical')
plt.rcParams['savefig.dpi']=300
plt.savefig(
    f"../../reports/figures/Number of Images for Each Label.png", bbox_inches='tight')
plt.show()
