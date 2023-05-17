# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 08:52:32 2022

@author: ING57
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def ish5dataset(item):
	return isinstance(item, h5py.Dataset)


def h5printR(item, leading = ''):
	for key in item:
		if ish5dataset(item[key]):
			print(leading + key + ': ' + str(item[key].shape))
		else:
			print(leading + key)
			h5printR(item[key], leading + '  ')

# Print structure of a .doric file            
def h5print(filename):
	with h5py.File(filename, 'r') as h:
		print(filename)
		h5printR(h, '  ')


def h5read(filename,where):
	data = []
	with h5py.File(filename, 'r') as h:
		item = h
		for w in where:
			if ish5dataset(item[w]):
				data = np.array(item[w])
				DataInfo = {atrib: item[w].attrs[atrib] for atrib in item[w].attrs}
			else:
				item = item[w]
	
	return data, DataInfo


def h5getDatasetR(item, leading = ''):
	r = []
	for key in item:
		# First have to check if the next layer is a dataset or not
		firstkey = list(item[key].keys())[0]
		if ish5dataset(item[key][firstkey]):
			r = r+[{'Name':leading+'_'+key, 'Data':
												[{'Name': k, 'Data': np.array(item[key][k]),
													'DataInfo': {atrib: item[key][k].attrs[atrib] for atrib in item[key][k].attrs}} for k in item[key]]}]
		else:
			r = r+h5getDatasetR(item[key], leading + '_' + key)
	
	return r


# Extact Data from a doric file
def ExtractDataAcquisition(filename):
	with h5py.File(filename, 'r') as h:
		#print(filename)
		return h5getDatasetR(h['DataAcquisition'],filename)


def extract_data_Ch2_ThreeSeries(folder, file_name):
	data =[]
	series = ['Series0001', 'Series0002', 'Series0003']
	for ser in series:
		dio01 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO01'])[0]
		dio02 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO02'])[0]
		dio03 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO03'])[0]
		dio04 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO04'])[0]
		dio_time = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','Time'])[0]
		infusion = find_event(dio02, dio_time)
		front    = find_event(dio03, dio_time)
		back     = find_event(dio04, dio_time)
		leverRetraction, leverInsertion = detect_edges(dio01, dio_time)
		# Load from Channel 2: AOUT01 is reference, AOUT02 is signal
		raw_reference = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'AIN02xAOUT01-LockIn','Values'])[0]
		raw_signal    = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'AIN02xAOUT02-LockIn','Values'])[0]
		# Load processed data
		signal        = h5read(folder + file_name + '_DFF.doric',['DataProcessed','FPConsole','DFFSignals',ser,'AIN02xAOUT02-LockIn','Values'])[0]
		time          = h5read(folder + file_name + '_DFF.doric',['DataProcessed','FPConsole','DFFSignals',ser,'AIN02xAOUT02-LockIn','Time'])[0]

		data_series = {'infusion': infusion, 'front': front, 'back': back, 'leverInsertion': leverInsertion, 'leverRetraction': leverRetraction, 'raw_reference': raw_reference,
					'raw_signal': raw_signal, 'signal': signal, 'time': time}
		data.append(data_series)

	all_data = []
	for i in range(len(data)):
		if i == 0:
			all_data = data[i]
		else:
			for key in data[i]:
				all_data[key] = np.concatenate((all_data[key], data[i][key]))
	return data, all_data

def extract_data_Ch4_ThreeSeries(folder, file_name):
	data =[]
	series = ['Series0001', 'Series0002', 'Series0003']
	for ser in series:
		dio01 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO01'])[0]
		dio02 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO02'])[0]
		dio03 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO03'])[0]
		dio04 = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','DIO04'])[0]
		dio_time = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'DigitalIO','Time'])[0]
		infusion = find_event(dio02, dio_time)
		front    = find_event(dio03, dio_time)
		back     = find_event(dio04, dio_time)
		leverRetraction, leverInsertion = detect_edges(dio01, dio_time)
		# Load from Channel 2: AOUT01 is reference, AOUT02 is signal
		raw_reference = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'AIN04xAOUT01-LockIn','Values'])[0]
		raw_signal    = h5read(folder + file_name + '.doric',['DataAcquisition','FPConsole','Signals',ser,'AIN04xAOUT02-LockIn','Values'])[0]
		# Load processed data
		signal        = h5read(folder + file_name + '_DFF1.doric',['DataProcessed','FPConsole','DFFSignals1',ser,'AIN04xAOUT02-LockIn','Values'])[0]
		time          = h5read(folder + file_name + '_DFF1.doric',['DataProcessed','FPConsole','DFFSignals1',ser,'AIN04xAOUT02-LockIn','Time'])[0]

		data_series = {'infusion': infusion, 'front': front, 'back': back, 'leverInsertion': leverInsertion, 'leverRetraction': leverRetraction, 'raw_reference': raw_reference,
					'raw_signal': raw_signal, 'signal': signal, 'time': time}
		data.append(data_series)
	all_data = []
	for i in range(len(data)):
		if i == 0:
			all_data = data[i]
		else:
			for key in data[i]:
				all_data[key] = np.concatenate((all_data[key], data[i][key]))
	return data, all_data


def find_event(signal, time):
    state = np.where(signal == 0)[0] # Med associates: 1 means Off, and 0 means On; Here 0 means there is something happing
    if len(state) == 0:
        event = []
    else:
        index = [0]
        for k in range(1, len(state)):
            if state[k] - state[k-1] > 1:
                index.append(k)
        event = time[state[index]]
    return event

def detect_edges(signal, time):
    last_state = signal[0]
    rising_edges = []
    falling_edges = []
    for i in range(1, len(signal)):
        if signal[i] > last_state:
            rising_edges.append(time[i])
        elif signal[i] < last_state:
            falling_edges.append(time[i])
        last_state = signal[i]
    return rising_edges, falling_edges

# Plot the PSTH of the signals
def psth_fb(data, time, event, pre, post, title_name):
    sample_rate = 1/np.mean(np.diff(time[0:1000]))
    sample_size = int((post-pre) * sample_rate)
    index = []
    psth_signal = np.zeros((sample_size, len(event)))
    psth_time   = np.zeros((sample_size, len(event)))
    for i in range(len(event)):
        index = np.where((time > event[i] + pre) & (time < event[i] + post))[0]
        if len(index) < sample_size:
            pass
        else:
            psth_time[:, i] = time[index[0:sample_size]] -  event[i]
            psth_signal[:,i] = data[index[0:sample_size]]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.mean(psth_time, 1), np.mean(psth_signal, 1), 'k')
    x = np.mean(psth_time, 1)
    y = np.mean(psth_signal, 1)
    e = np.std(psth_signal, axis=1)/np.sqrt(psth_signal.shape[1])
    plt.fill_between(x, y-e, y+e)
    plt.xlabel('Time (s)')
    plt.ylabel(r'z $\Delta$ F/F')
    plt.xlim([np.min(np.mean(psth_time, 1)), np.max(np.mean(psth_time, 1))])
    # ylim([-0.1, 0.1])
    plt.ylim([-1.5, 1.5])
    plt.tick_params(direction='out', length=5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplots_adjust(hspace=0.5)
    plt.title(title_name)

    plt.subplot(2, 1, 2)
    plt.imshow(psth_signal.T, aspect='auto', cmap='jet', extent=[np.min(np.mean(psth_time, 1)), np.max(np.mean(psth_time, 1)), 0,len(event)])
    # colorbar
    plt.xlabel('Time (s)')
    plt.ylabel('Trials')
    plt.tick_params(direction='out', length=5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return psth_time, psth_signal