import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

#import dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')

#drop rows containing missing values
df.dropna(axis = 0, how = 'any', subset = None, inplace = True)

# Convert Non-numeric data using one-hot encoding
df = pd.get_dummies(df,columns=['island','sex'])
df.head()

# Assing X and Y variables
X = df.drop('species', axis=1)
Y = df['species']

# Split data into test/train set (70/30 split) and shuffle
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True)

#Assign algorithm
model = LogisticRegression(solver = 'lbfgs', max_iter = 6000)

# Link algorithm to X and Y variables
model.fit(X_train, Y_train)

# Run algorithm on test data to make predictions
model_test = model.predict (X_test)

#Evaluate predictions
print(confusion_matrix(Y_test,model_test))
print(classification_report(Y_test,model_test))

# Data point to predict
penguin = [
	56, #bill_length_mm
	12.5, #bill_depth_mm
	120, #flipper_length_mm 
	3150, #body_mass_g
	0, #island_Biscoe    
	1, #island_Dream
	0, #island_Torgersen    
	0, #sex_Male
	1, #sex_Female
]

# Make prediction

feature_names_trained = model.feature_names_in_
penguin_df = pd.DataFrame([penguin], columns=feature_names_trained)

new_penguin = model.predict(penguin_df)
new_penguin

probabilidades = model.predict_proba(X_test)
print(probabilidades)
