#Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Import dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
del df['smoker']

# Convert non-numeric data using one-hot encoding
df = pd.get_dummies(df,columns=['time','day','sex'])
df.head()

# Exploratory data analysis: correlation and heatmap
df_corr = df.corr()
df_corr
#sns.heatmap(df_corr,annot= True,cmap='coolwarm')

# Exploratory data analysis: pairplot
#sns.pairplot(df)

# Assign X and Y variables
X = df.drop('tip', axis = 1)
Y = df['tip']

# Split data into test/train set (70/30) and shuffle
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True)


# Assign algorithm
model = LinearRegression()

# Link algorithm to X and Y variables
model.fit(X_train, Y_train)

# Find Y-intercept
model.intercept_

# Find X coefficient
model.coef_


#Models training and testing absolute error
mae_train = mean_absolute_error(Y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.2f" %mae_train)

mae_test = mean_absolute_error(Y_test, model.predict(X_test))
print("Training Set Mean Absolute Error: %.2f" %mae_test)

#Prediction
# Data point to predict
jamie = [
	40, #total_bill
	2, #size
	1, #time_dinner
	0, #time_lunch
	1, #day_fri
	0, #day_sat
	0, #day_sun
	0, #day_thur
	1, #sex_female
	0, #sex_male
]

# Extract trained model data labels (the ones it remembers)
feature_names_trained = model.feature_names_in_

# Create dataframe for prediction
jamie_df = pd.DataFrame([jamie], columns=feature_names_trained)

# Make prediction
jamie = model.predict(jamie_df)
jamie
