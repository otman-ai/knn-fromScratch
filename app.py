import numpy as np
import pandas as pd
from KNN import KNN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import subprocess

# Define the wget command as a list of arguments
wget_command = ["wget", "-O", "tele100.csv", "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv"]

# Execute the wget command
try:
    subprocess.run(wget_command, check=True)
    print("File downloaded successfully.")
except subprocess.CalledProcessError as e:
    print("Error:", e)

model = KNN(task='classification',K=5)
df = pd.read_csv('tele100.csv')
df.head()
X = df.drop('custcat',axis=1)
y = df['custcat']

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4)

y_pred =  [[model.predict(X_train,y_train.values, x_test)][0] for x_test in X_test]
y_pred

print("Test set accuracy: ", metrics.accuracy_score(y_pred,y_test.values))
