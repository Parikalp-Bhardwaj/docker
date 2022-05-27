import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


data = pd.read_csv("./Titanic.csv")
#print(data.head())

# ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']
#print(data.isnull().sum())

data.drop(data[["PassengerId","Name","Ticket"]],inplace=True,axis=1)
# print(data.head())
# print("shape ",data.shape)
# data = data.dropna(how='any')


pclass_sex = data.groupby(['Sex', 'Pclass']).median()['Age']
#print(pclass_sex)

# for pclass in range(1, 4):
#     for sex in ['female', 'male']:
#         print('Median age of Pclass {} {}s: {}'.format(pclass, sex, pclass_sex[sex][pclass]))
# print('Median age of all passengers: {}'.format(data['Age'].median()))

data["Age"] = data.groupby(["Sex","Pclass"])["Age"].apply(lambda x:x.fillna(x.median()))


data.drop("Cabin",axis=1,inplace=True)

data = data.dropna(how='any')



classSex = {idx:val for val,idx in enumerate(np.unique(data["Sex"]))}
data["Sex"] = data["Sex"].map(classSex)


classEmbarked = {idx:val for val,idx in enumerate(np.unique(data["Embarked"]))}
data["Embarked"] = data["Embarked"].map(classEmbarked)



x = data.drop("Survived",axis=1).astype(np.float64).values
y = data["Survived"].values

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

logistic = LogisticRegression() 
rfe = RFE(estimator=logistic,n_features_to_select=2)
pipeline = Pipeline(steps=[('s',rfe),('m',logistic)])

pipeline.fit(X_train,y_train)

print("Score ",pipeline.score(X_test,y_test))

# pclass = int(input("Enter Ticket class 1 : 1st, 2 : 2nd, 3 : 3rd :==> "))
# sex = int(input(f"Enter Sex {classSex} :==> "))
# age = int(input("Enter Age :==> "))
# sibSp = int(input("Enter family relations in this way siblings / spouses aboard the '1' for yes '0' for no :==> "))
# parch = int(input("Enter family relations in this way parents / children aboard the '1' for yes '0' for no :==> "))
# fare = int(input("Enter Passenger fare :==> "))
# embarked = int(input(f"Enter Port of Embarkation {classEmbarked}: ==>"))

# df = np.array([[pclass,sex,age,sibSp,parch,fare,embarked]],dtype=np.float64)



class Items(BaseModel):
    pclass: int
    sex: int
    age: int
    sibSp: int
    parch: int
    fare: int
    embarked: int
    


@app.post("/create-api")
def pred(item:Items):
    df = np.array([[item.pclass,item.sex,item.age,item.sibSp,item.parch,item.fare,item.embarked]],dtype=np.float64)
    predict = pipeline.predict(df)[0]
    if predict == 1:
        return "Survival"
    
    return "Not Survival"

if __name__=="__main__":
    uvicorn.run(app,port=8000,host="0.0.0.0")