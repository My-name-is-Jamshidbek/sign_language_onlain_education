import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open(r'../create_dataset/dataset.pickle', 'rb'))
print(type(data_dict['data']))
s1 = len(data_dict['data'][0])
j = 0
m = 0
nm = []
nl = []
print(data_dict['data'])
for i in data_dict['data']:
    s = len(i)
    if s != s1:
        print(s,s1, data_dict['labels'][j])
        m+=1
    else:
        nm.append(i)
        nl.append(data_dict['labels'][j])
    j+=1
print(m)

data = np.asarray(nm)
labels = np.asarray(nl)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open(r'model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
