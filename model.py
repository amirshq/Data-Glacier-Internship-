import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn import preprocessing

#load csv file
df = pd.read_csv('ENB2012_data.csv')
print(df.head())

#select dependant and independant variables
X = df[['X1','X2','X3','X4','X5','X6']]
y = df[['Y1']]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test )

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
lab = preprocessing.LabelEncoder()
y_train = y_train.values.ravel()
y_transformed = lab.fit_transform(y_train)

classifier.fit(X_train, y_transformed)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

