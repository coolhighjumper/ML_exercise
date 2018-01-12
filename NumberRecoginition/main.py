'''
@author: Chiang
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP

x = pd.read_csv('xx.csv',header=None)
y = pd.read_csv('yy.csv',header=None)
x=np.array(x)
y=np.array(y)

x_train=x[0:1000]
y_train=y[0:1000]
x_validation=x[40000:]
y_validation=y[40000:]

ppn=MLP(eta=0.1,n_iter=10,n_hidden=120,n_out_put=10)
ppn.fit(x_train, y_train)
print(ppn.errors_)

count=0
for i in range(x_validation.shape[0]):
    hidden_input=np.dot(x_validation[i],ppn.w_1)[:,None]+ppn.b_1
    hidden_output=1/(1+np.exp(-hidden_input))



    z=np.dot(np.transpose(hidden_output),ppn.w_2)
    z=np.transpose(z)+ppn.b_2
    #soft-max
    output=(np.exp(z)/np.sum(np.exp(z),axis=0))
    #index of max
    ans=np.zeros(ppn.n_out_put)

    ans[np.where(output==output.max())[0][0]]=1


    flag=True;
    for j in range(ppn.n_out_put):
        if ans[j]!=y_validation[i][j]:
            flag=False
            break;
        else:
            continue;
    if flag==False:
        count+=1
print(x_validation.shape)
print(count)


test = pd.read_csv('test_mod.csv',header=None)
test=np.array(test)

result=np.zeros(test.shape[0])

for i in range(test.shape[0]):
    hidden_input=np.dot(test[i],ppn.w_1)[:,None]+ppn.b_1
    hidden_output=1/(1+np.exp(-hidden_input))



    z=np.dot(np.transpose(hidden_output),ppn.w_2)
    z=np.transpose(z)+ppn.b_2

    #soft-max
    output=(np.exp(z)/np.sum(np.exp(z),axis=0))

    #index of max
    result[i]=np.where(output==output.max())[0][0]

print(result.shape)
np.savetxt('result.csv',result , delimiter=',',fmt='%i')




plt.plot(range(1,len(ppn.errors_)+1),ppn.entropy_,marker='o')
plt.xlabel('Epochs')
#plt.ylabel('Number of misclassifications')
plt.ylabel('CrossEntropy')
plt.show()
