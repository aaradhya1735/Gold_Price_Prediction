import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('gld_price_data.csv')

# number of rows and columns
gold_data.shape

# getting some basic informations about the data
gold_data.info()

# getting the statistical measures of the data
gold_data.describe()

X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.05, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)

# training the model
regressor.fit(X_train,Y_train)

# prediction on Test Data
test_data_prediction = regressor.predict(X_test)
print("Predicted Gold Prices", test_data_prediction )

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

Y_test = list(Y_test)

plt.plot(Y_test, color='red', label = 'Actual Value')
plt.plot(test_data_prediction, color='blue', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()