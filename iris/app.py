import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

species_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


st.set_page_config(
    page_title="Iris Species Classification",
    layout="wide"
)

st.title("Iris Species Classification Dashboard")

st.markdown("""
This dashboard predicts Iris flower species using a Random Forest classification model.
""")


st.header("Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")


st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    2.0,
    4.5,
    3.5
)

petal_length = st.sidebar.slider(
    "Petal Length",
    1.0,
    7.0,
    1.4
)

petal_width = st.sidebar.slider(
    "Petal Width",
    0.1,
    2.5,
    0.2
)


new_sample = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

new_sample_scaled = scaler.transform(new_sample)

prediction = model.predict(new_sample_scaled)

predicted_species = species_names[prediction[0]]

st.success(f"Predicted Species: {predicted_species}")


st.header("3D Iris Dataset Visualization")

plot_df = X.copy()

plot_df["species"] = [
    species_names[i]
    for i in y
]

plot_df.loc[len(plot_df)] = [
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
    "New Sample"
]

fig = px.scatter_3d(
    plot_df,
    x='sepal length (cm)',
    y='petal length (cm)',
    z='petal width (cm)',
    color='species',
    title='3D Scatter Plot'
)

st.plotly_chart(fig, use_container_width=True)


st.header("Feature Distribution")

selected_feature = st.selectbox(
    "Select a Feature",
    X.columns
)

hist_fig = px.histogram(
    plot_df,
    x=selected_feature,
    color='species',
    barmode='overlay'
)

st.plotly_chart(hist_fig, use_container_width=True)


st.header("Scatter Matrix")

scatter_matrix = px.scatter_matrix(
    plot_df,
    dimensions=X.columns,
    color='species'
)

st.plotly_chart(scatter_matrix, use_container_width=True)