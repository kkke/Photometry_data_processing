#%% Load necessary module
import os
import doric as dr
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#%% Load your data
# Load your data
folder = 'H:/Data/FB_data/Dopamine/SA74/042523/'
file_name = 'SA74_Ch2_NAc_Ch4_DLS_042523_0000'
region = 'NAc'
data, all_data = dr.extract_data_Ch2_ThreeSeries(folder, file_name)

#%%
# save the processed file;
savefile = folder + file_name + '_processed_' + region
with open(savefile + '.pkl', 'wb') as f:
    pickle.dump([data, all_data], f)
summaryfolder = 'H:/Data/FB_data/Dopamine/SA74/'
with open(summaryfolder + file_name + '_processed_' + region + '.pkl', 'wb') as f:
    pickle.dump([data, all_data], f)
#%%
# Load processed data
folder = 'H:/Data/FB_data/Dopamine/SA77/051823/' # Modify it depending on where your file is located
# File name
file_name = 'SA77_NAc_051823_0000'   # Change to your data file
savefile = folder + file_name + '_processed_NAc.pkl'
with open(savefile, 'rb') as f:
    data, all_data = pickle.load(f)

#%%PSTH aligned to drug infusion
psth_time, psth_signal, fig = dr.psth_fb(all_data['signal'], all_data['time'], all_data['infusion'], -5, 50, 'Drug Infusion')
fig.savefig(folder + file_name + region + '_infusion.pdf')
# %%
#%%PSTH aligned to LeverInsertion
psth_time, psth_signal, fig = dr.psth_fb(all_data['signal'], all_data['time'], all_data['leverInsertion'], -5, 10, 'Drug Infusion')
fig.savefig(folder + file_name + region + '_leverInsertion.pdf')
# %%
