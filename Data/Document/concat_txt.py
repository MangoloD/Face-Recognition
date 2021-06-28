import pandas as pd

df = pd.read_table("list_bbox_celeba.txt", sep="\s+", header=1, low_memory=False)
dd = pd.read_table("list_landmarks_celeba.txt", sep="\s+", header=1, low_memory=False)
dd = dd.reset_index().rename(columns={'index': 'image_id'})
data = pd.merge(df, dd, how="left", on="image_id")
data = data[["image_id", "x_1", "y_1", "width", "height", "lefteye_x", "lefteye_y", "righteye_x", "righteye_y",
             "nose_x", "nose_y", "leftmouth_x", "leftmouth_y", "rightmouth_x", "rightmouth_y"]]
data.to_csv("data.csv", index=False)
