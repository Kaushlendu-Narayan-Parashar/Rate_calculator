import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('rate.csv')

#fairness rated on fair-an-lovely scale
plt.scatter(df['Fairness'], df['Rate'])
plt.xlabel('Fairness')
plt.ylabel('Rate')
#plt.show()

l_reg = linear_model.LinearRegression()
l_reg.fit(df[['Fairness']], df.Rate)

#print(l_reg.predict([[0]]))
print(l_reg.coef_)
print(l_reg.intercept_)
