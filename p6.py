
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

st.title("Titanic Survival Prediction")

uploaded_file = st.file_uploader("Upload titanic-dataset.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.head()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.dropna(subset=['Embarked'])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
    y = df['Survived']
    df.head()

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(classification_report(y_test,y_pred))
    print("Accuracy:", round (accuracy_score (y_test,y_pred),3))
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    st.text(classification_report(y_test,y_pred))
    st.write("Accuracy:", round (accuracy_score (y_test,y_pred),3))

    st.pyplot(plt.gcf())
