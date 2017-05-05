import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('input here the destination of any csv file which has single input feature and a label')
Single_feature = data.iloc[:, :-1].values #Here the label is last column and the feature is the first column
Label = data.iloc[:, 1].values

# Here the dataset is splitted into training and testing set. test_size=0.2 specifies that 20% of dataset will be testing data.
from sklearn.cross_validation import train_test_split
Feature_train, Feature_test, Label_train, Label_test = train_test_split(Single_feature, Label, test_size = 0.2, random_state = 0)


# Simple linear regression model on training set
from sklearn.linear_model import LinearRegression
reg_model=LinearRegression()
reg_model.fit(Feature_train, Label_train)

#By fitting the linear regression model to the training set, we made our machine 
#learn the correlation between the features and the label of training set.

#Now we use this model, which has learned correlations between the dependent 
#and independent variables, to predict the test set results.

Label_predicted = reg_model.predict(Feature_test)

#Now, plotting the graphs
plt.scatter(Feature_train, Label_train, color = 'pink')
plt.plot(Feature_train, reg_model.predict(Feature_train), color = 'green')
plt.title('Correlation in Training Set')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.show()

plt.scatter(Feature_test, Label_test, color = 'pink')
plt.plot(Feature_train, regressor.predict(Feature_train), color = 'green')
plt.title('Correlation in Test set')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.show()
