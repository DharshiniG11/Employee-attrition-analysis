import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.core.reshape.reshape import unstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("E:/ibm hr analytics.csv")
sns.countplot(x="Attrition",hue="Gender",data=df,palette="Blues")
plt.title("Employee Attrition Analysis")
plt.show()

by_dept = df[df["Attrition"]=="Yes"].groupby(["Department","Gender"]).size().unstack()
by_dept.plot(kind="bar",stacked=True,color=["blue","orange"],figsize=(10,6))
plt.title("department wise gender attrition")
plt.xlabel("department")
plt.ylabel("no of employees")
plt.legend(title="Gender")
plt.show()

sns.boxplot(x="Attrition",y="MonthlyIncome",data=df)
plt.title("does income level affect attrition rate")
plt.show()

sns.countplot(x="WorkLifeBalance",hue="Attrition",data=df)
plt.title("work-life balance v/s attrition")
plt.show()




categories = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
for col in categories:
    df[col]= LabelEncoder().fit_transform(df[col])
    if "Over18" in df.columns:
        df = df.drop(columns=["Over18"])  # Drop only if it exists

print(df.head())
df["Attrition"]  = LabelEncoder().fit_transform(df["Attrition"])
x = df.drop("Attrition",axis=1)
y=df["Attrition"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)
print(x_train.dtypes)
print(x_train.select_dtypes(include=['object']).head())  # Show only text columns
print(x_test)
print(y_train)
print(y_test)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"🔹 Random Forest Accuracy: {accuracy_rf:.2f}")
