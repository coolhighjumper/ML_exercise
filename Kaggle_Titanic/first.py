import numpy as np
import pandas as pd
dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")
dfTrain.head(1).info()
dfTest.head(1).info()
dfTrain.set_index(['PassengerId'],inplace=True)
dfTest.set_index(['PassengerId'],inplace=True)

import matplotlib.pyplot as plt

dfTrain.groupby('Pclass').Survived.mean().plot(kind='bar')
plt.show()
