import pandas as pd
df=pd.read_csv("game_knn.csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
label=le.fit_transform(df["Gender"])
df=df.drop('Gender',axis="columns") 
df.insert(2,"Gender",label) # Inserting gender values 
x=df.iloc[:,1:3].values
y=df["Sport"]
y=le.fit_transform(y)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
classifier=KNeighborsClassifier(n_neighbors=3,metric="euclidean")
classifier.fit(x,y)
y_pred=classifier.predict(x)
cm=confusion_matrix(y,y_pred)
accuracy_score(y,y_pred)
classifier.predict([[5,0]])
