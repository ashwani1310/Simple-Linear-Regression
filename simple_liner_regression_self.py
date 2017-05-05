import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('Salary_Data.csv')
Experience = data.iloc[:, :-1].values
Salary = data.iloc[:, 1].values


from sklearn.cross_validation import train_test_split
Exp_train, Exp_test, Sal_train, Sal_test = train_test_split(Experience, Salary, test_size = 0.2, random_state = 0)


# Simple linear regression model on training set
from sklearn.linear_model import LinearRegression
reg_model=LinearRegression()
reg_model.fit(Exp_train, Sal_train)

#By fitting the linear regression model to the training set, we made our machine 
#learn the correlation between the features and the label of training set.

#Now we use this model, which has learned correlations between the dependent 
#and independent variables, to predict the test set results.

Sal_predicted = reg_model.predict(Exp_test)

#Now, plotting the graphs
plt.scatter(Exp_train, Sal_train, color = 'pink')
plt.plot(Exp_train, reg_model.predict(Exp_train), color = 'green')
plt.title('Correlation in Training Set')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(Exp_test, Sal_test, color = 'pink')
plt.plot(Exp_train, regressor.predict(Exp_train), color = 'green')
plt.title('Correlation in Test set')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()