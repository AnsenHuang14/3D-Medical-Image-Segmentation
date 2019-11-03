import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import os
import json
p = lambda i: "../data/data/{}".format(i) 
address = ['./data/data/{}/'.format(i) for i in range(131) if os.path.exists(p(i))]
number_of_instance = [len(os.listdir('../data/data/{}/gt/'.format(i))) for i in range(131) if os.path.exists(p(i)) ]
start_list, end_list = list(),list()
for i in range(131):
	if os.path.exists(p(i)): 
		with open('../data/data/{}/indice.txt'.format(i), 'r') as file:
			start, end = list(map(int, file.read().split(',')))
			start_list.append(start)
			end_list.append(end)
data = pd.DataFrame({'address':address,'number_of_instance':number_of_instance,'start':start_list,'end':end_list})
# print(address,number_of_instance)
data = shuffle(data)
data = data.reset_index(drop=True)
# print(data)
train = data
test = data
print("Training set shape:",train.shape,"Testing set shape",test.shape)
train.to_json("../data/Train.json",orient='records',lines=False)
test.to_json("../data/Test.json",orient='records',lines=False)

# with open(os.path.join('./data/Train.json'), 'r') as j:
# 	instance = json.load(j)
# 	print(instance)