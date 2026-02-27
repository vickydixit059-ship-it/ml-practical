import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
st.title("KNN Weather Classification")
X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# User input
temp = st.slider("Temperature (°C)", 20, 40, 26)
humidity = st.slider("Humidity (%)", 50, 90, 78)

new_point = np.array([[temp, humidity]])
prediction = knn.predict(new_point)[0]
fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Sunny", s=100)
ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Rainy", s=100)
ax.scatter(temp, humidity, marker="*", s=300, color="red", label="New Prediction")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
if prediction == 0:
    st.success("Predicted Weather: Sunny ☀️")
else:
    st.info("Predicted Weather: Rainy 🌧️")

