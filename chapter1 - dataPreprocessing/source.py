# importing libraries
import numpy as np
import pandas as pd

# importing dataset
dataset=pd.read_csv('data.csv') # method to read data file, delimiter - comma
X=dataset.iloc[:,:-1].values# independet variable matrix, iloc - imports columns, all except last
y=dataset.iloc[:,3].values# dependent variable vector, importing all rows, 3 columns

# Handling missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean') # replaces nan values with a place holder
imputer=imputer.fit(X[:,1:3]) # passing the location of the missing data (cell C6)
X[:,1:3]=imputer.transform(X[:,1:3]) # ensures continuity of data by transforming to a processable stage. Usually inserts mean value of column as placeholder

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder() # converts string data to numerical data
X[:,0]=labelencoder_X.fit_transform(X[:,0]) # encoding all features of column 1 (country names) into numbers 0, 1, 2 for priority 2 > 1 > 0, which is a nonsensical relation atm

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough') # replaces column variables with 1s and 0s
X = ct.fit_transform(X)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Splitting dataset into training set and testing test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) # Seeration of data for training ANN and testing afterwards, 20% seperation (0.2 test size), random state = 0 (random seed)

# feature scaling (standardisation technique here, not normalisation)
'''
Standardisation - subtraction of mean from matrix values and dividing by standard deviance (gaussian distributions)
Normalisation - dividing matrix values by length (non-gaussian distributions)
'''
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler() # equivalent to graph
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
