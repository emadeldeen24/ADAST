import os
from scipy.signal import resample
import numpy as np

data_path  = "E:/shhs1"
output_dir = "E:/shhs1_resampled"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)




files = []
for i in os.listdir(data_path):
    files.append(os.path.join(data_path, i))


for np_file in files:
    e = np.load(np_file)
    x = e["x"].squeeze()
    y = e["y"]
    sampling_rate = e["fs"]

    new_x = []

    for row in range(x.shape[0]):
        row_data = x[row,:]
        new_row_data = resample(row_data, 3000)
        new_x.append(new_row_data) 
    new_x = np.array(new_x)


    # Saving as numpy files
    # print(file, x_values.shape[0], y_values.shape[0])
    filename = os.path.basename(np_file).replace(".", "_resampled.")
    save_dict = {
        "x": new_x,
        "y": y,
        "fs": 100
    }
    np.savez(os.path.join(output_dir, filename), **save_dict)
#    print(" ---------- Done ---------")

print("done")
