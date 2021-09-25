import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

np.random.seed(0)
x=np.random.rand(100,1)
y=2+3*x+np.random.rand(100,1)

plt.scatter(x,y,s=10)
regression_model=LinearRegression()
regression_model.fit(x,y)

y_predicted=regression_model.predict(x)


#evaluation
rmse=mean_squared_error(y,y_predicted)
r2=r2_score(y,y_predicted)

print("Slope" ,regression_model.coef_)
print("Intercept: ",regression_model.intercept_)
print("RMS error", rmse)
print('R2 score', r2)

#plotting predicted values
plt.scatter(x,y,s=10)

plt.plot(x,y_predicted,color="r")
plt.show()