from datetime import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
from itertools import cycle

import glob

import ast

from typing import List

pd.set_option('display.max_columns', 20)

def str_to_list(x):
    return sum(ast.literal_eval(x))

def smooth_data(scalars: List[float], weight: float) -> List[float]:
    """Tensorboard smoothing function to smooth noisy training data

    :param scalars: data points to smooth
    :type scalars: List[float]
    :param weight: Exponential Moving Average weight in 0-1
    :type weight: float
    :return: smoothed data points
    :rtype: List[float]
    """
    assert weight >= 0 and weight <= 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def plot_info(combined, cat):
    names = combined[0].columns.tolist()
    names = [name for name in names if type(name) == str]


    fig, axs = plt.subplots(3,4,figsize=(9, 9))
    axs = np.ndarray.flatten(axs).tolist()
    fig.delaxes(axs[-1])

    for i, ax in enumerate(axs[:len(names)]):
        for data in combined:
            y = data[names[i]].to_numpy()
            ax.plot(y, alpha=0.1)
            
        for data in combined:
            y = data[names[i]].to_numpy()
            ax.plot(smooth_data(y, 0.99), alpha=0.8)
            
        ax.title.set_text(names[i].replace("_"," ").title())  

    fig.suptitle(cat.title(), fontsize=20) 
    fig.tight_layout()

    plt.show()

def process_data(folder_path, run_type):
    files = glob.glob(folder_path + "*")
    files = pd.DataFrame(files)

    files = pd.concat([files, files[0].str.split("/|\\.|\\_", expand=True).iloc[:,4:(len(files.columns)+4+2)]], axis=1)
    files = files.sort_values([4, 5, 6], ascending=[False, True, True])



    files = files[files[4] == run_type]

    dfs = [pd.read_csv(file_) for file_ in files[0].tolist()]


    names = dfs[0].dtypes[dfs[0].dtypes == object].index.to_list()

    for i in range(len(dfs)):
        names = dfs[i].dtypes[dfs[i].dtypes == object].index.to_list()
        for name in names:
            dfs[i][name] = dfs[i][name].apply(np.vectorize(str_to_list))
        
        dfs[i].name = files.iloc[i,0]
        dfs[i]["file_name"] = files.iloc[i,0]
        dfs[i] = pd.merge(dfs[i], files, how="left", left_on=["file_name"], right_on = [0]).drop(["file_name", 0],1)

    test = files.duplicated(subset=[5]).tolist()
    inds = [i for i, x in enumerate(test) if not x]
    inds = [(inds[i],inds[i+1]) if i+1 != len(inds) else (inds[i], None) for i in range(len(inds))]

    combine = [dfs[tup[0]:tup[1]] for tup in inds]
    combined = [pd.concat(comb) for comb in combine]
    
    return combined

root_folder = "./outputs/"
network = "big-intersection/"

folder_path = root_folder + network

run_type = "train"

combined = process_data(folder_path, run_type)

test = pd.concat(combined)
pair_res = test.groupby([6,5]).agg(['mean']).reset_index()
run_res = test.groupby([6]).agg(['mean']).reset_index()


plot_info(combined, run_type)


# time_steps = dfs[0].iloc[:,0].values
# time_steps = np.expand_dims(time_steps,0)
# waits = np.array([df.iloc[:,-1].values for df in dfs])
# runs = np.array(list(range(len(files))))
# runs = np.expand_dims(runs,0)
# fig = plt.figure()
# ax = plt.axes(projection = "3d")

# ax.plot_surface(time_steps, runs.T, waits, rstride=1, cstride=1)
# ax.invert_xaxis()
# plt.show()

print("tru")