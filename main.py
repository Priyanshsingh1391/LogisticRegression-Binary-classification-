import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('insurance_data.csv')
print(df.head())
plt.scatter(df.age,df.bought_insurance, color='red')
plt.show()

X_train, X_test, y_train, y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print(prediction)

# to check the accuracy of the model
acc= model.score(X_test,y_test)

print(acc)

#to predict
pred = model.predict(X_test)
print(pred)

