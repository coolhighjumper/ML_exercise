'''
@author: Chiang
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP

x = pd.read_csv('train.csv')
y = pd.read_csv('y_train.csv')
del x['ID']
del y['ID']
x=np.array(x)
y=np.array(y)
print(x.shape)
print(y.shape)
x_test=x[0:350]
x_verify=x[224:]
y_test=y[0:350]
y_verify=y[224:]

ppn=MLP(eta=0.5,n_iter=30,n_hidden=5)
ppn.fit(x_test, y_test)
print(ppn.errors_)


test = pd.read_csv('test.csv')
del test['ID']
test=np.array(test)
ans=[]
count=0
for i in range(x_verify.shape[0]):
    data_input=np.append(x_verify[i],1)
    hidden_input=np.dot(np.array([data_input]),ppn.w_1)
    hidden_output=1/(1+np.exp(-hidden_input))
    hidden_output=np.append(hidden_output,1)
    output=1/(1+np.exp(-np.dot(hidden_output,ppn.w_2)))
    #print(output)
    if np.round(output)!=y_verify[i]:
        count+=1

for i in range(test.shape[0]):
    data_input=np.append(test[i],1)
    hidden_input=np.dot(np.array([data_input]),ppn.w_1)
    hidden_output=1/(1+np.exp(-hidden_input))
    hidden_output=np.append(hidden_output,1)
    output=1/(1+np.exp(-np.dot(hidden_output,ppn.w_2)))
    #print(output)
    if np.round(output)==0:
        ans.append(0)
    else:
        ans.append(1)

print(count)
np.savetxt('ans.csv',ans , delimiter=',')

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()


