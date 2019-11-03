import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import os
import json
root_folder_path = '../data/2019Sep/'
tmp_address = ['../data/2019Sep/{}/'.format(folder) for folder in os.listdir(root_folder_path)]
address = ['./data/2019Sep/{}/'.format(folder) for folder in os.listdir(root_folder_path)]
number_of_instance = [len(os.listdir(ad+"gt/")) for ad in  tmp_address]
# print(address,number_of_instance)

data = pd.DataFrame({'address':address,'number_of_instance':number_of_instance})

data = shuffle(data)
data = data.reset_index(drop=True)
# print(data)
train = data
test = data
print("Training set shape:",train.shape,"Testing set shape",test.shape)
train.to_json("../data/DCM_Train.json",orient='records',lines=False)
test.to_json("../data/DCM_Test.json",orient='records',lines=False)
