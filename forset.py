import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].values
Y=data.iloc[:,2].values
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,Y)
y_pred=regressor.predict(6.5)
x_grid=np.arange(min(X),max(X),0.01)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('decision tree')
plt.show()