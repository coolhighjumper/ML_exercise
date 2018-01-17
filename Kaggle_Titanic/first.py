import numpy as np
import pandas as pd
import sys
#specify absolute path(vsCode has the issue can't open the file)
dir=sys.path[0]
dfTrain = pd.read_csv(dir+"/train.csv")
dfTest = pd.read_csv(dir+"/test.csv")
dfTrain.head(1).info()
dfTest.head(1).info()
dfTrain.set_index(['PassengerId'],inplace=True)
dfTest.set_index(['PassengerId'],inplace=True)

import matplotlib.pyplot as plt

dfTrain.groupby('Pclass').Survived.mean().plot(kind='bar')
plt.show()
