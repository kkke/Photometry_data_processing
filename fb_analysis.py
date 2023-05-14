# Load necessary module
import os
import doric as dr
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# Load your data
# Folder with your files
folder = 'H:/Data/FB_data/Dopamine/SA76/050823/' # Modify it depending on where your file is located
# File name
file_name = 'SA76_Ch2_NAc_Ch4_DLS_050823_0001'   # Change to your data file
data, all_data = dr.extract_data_Ch2_ThreeSeries(folder, file_name)

# Plot the raw data
fig = plt.figure(figsize = (8, 4))
plt.scatter(all_data['infusion'], np.ones_like(all_data['infusion']))
plt.plot(all_data['time'], all_data['signal'], 'red')
plt.show()

#PSTH aligned to drug infusion
psth_time, psth_signal = dr.psth_fb(all_data['signal'], all_data['time'], all_data['infusion'], -5, 60)

