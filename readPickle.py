import pickle 
import numpy as np

# read pickle file 
with open("keypoints.pkl", "rb") as f:
    data=pickle.load(f)

print("Loaded shape:", data.shape)
print(data)

print("type:", type(data))
print("Is numpy.ndarray?", isinstance(data, np.ndarray))
print("Shape:", data.shape)

