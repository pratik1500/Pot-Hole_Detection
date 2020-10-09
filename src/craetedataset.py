import pandas as pd
import numpy as np
import os

target = []
img_path = []
for dirname, _, filenames in os.walk(r'Data\normal'):
    for filename in filenames:
        img_path.append(os.path.join(dirname, filename))
        target.append(0)
for dirname, _, filenames in os.walk(r'Data\potholes'):
    for filename in filenames:
        img_path.append(os.path.join(dirname, filename))
        target.append(1)
print(len(target), len(img_path))

potholes_data = {"data_dir": img_path, "values": target}
df = pd.DataFrame(potholes_data, columns=['data_dir', 'values'])
print(df)
df.to_csv('pot_hole_data.csv', index=True)
df = pd.read_csv('pot_hole_data.csv')
print(df.head())
