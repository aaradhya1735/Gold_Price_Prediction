import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('gld_price_data.csv')

# Getting the number of rows and columns in the DataFrame
gold_data.shape

# Getting some basic information about the data (data types, non-null counts, etc.)
gold_data.info()

# Getting the statistical measures of the data (count, mean, std, min, quartiles, max)
gold_data.describe()

# Extracting the feature variables (X) and target variable (Y)
# Dropping 'Date' and 'GLD' columns from X as they are not used for prediction
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

# Splitting the data into training and testing sets with a test size of 5%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.05, random_state=2)

# Creating a RandomForestRegressor model with 100 decision trees
regressor = RandomForestRegressor(n_estimators=100)

# Training the model on the training data
regressor.fit(X_train, Y_train)

# Making predictions on the test data
test_data_prediction = regressor.predict(X_test)
print("Predicted Gold Prices", test_data_prediction)

# Calculating the R-squared error to evaluate the model's performance
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

# Converting Y_test to a list for plotting purposes
Y_test = list(Y_test)

# Plotting the actual values (Y_test) in red and the predicted values in blue
plt.plot(Y_test, color='red', label='Actual Value')
plt.plot(test_data_prediction, color='blue', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
