# to visualise feel free to go to this link:
# https://aboudileila-streamlitfirstproject-iris-dataset-predictor-txzgdx.streamlit.app/
import streamlit as st
from PIL import Image
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import plotly.express as px
import statistics

icon_img = Image.open('Iris.jpg')
st.set_page_config(
    page_title='Iris Dataset Analysis',
    page_icon=icon_img,
    layout="wide",
    initial_sidebar_state='collapsed'
)

iris = load_iris()
df = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})

X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=20
                                                    )
clf = RandomForestClassifier(n_estimators=4, criterion='gini')
model = clf.fit(x_train, y_train)


def refresh_page():
    st.experimental_rerun()


if st.button('Actualiser', key='refresh'):
    refresh_page()


def separation(titre1=None, titre2=None, color='cyan'):
    st.text('')
    if titre1:
        st.markdown(
            f"<h1 style='font-family: Lucida Console;text-align: center; color: {color};'>{titre1}</h1>",
            unsafe_allow_html=True
        )
    if titre2:
        st.markdown(
            f"<h4 style='font-family: Lucida Console;text-align: center; color: {color};'>{titre2}</h4>",
            unsafe_allow_html=True
        )
    st.text('')


separation('Iris Dataset Analysis') #This is a markdown used as a title
# st.title("Iris Dataset Analysis") #This is the title in a standard way
st.header("_:blue[Exploring the Iris Dataset 💐 🌺]_")

sepal_length = st.slider("Sepal Length",
                         float(df['sepal length'].min()),
                         float(df['sepal length'].max()),
                         float(df['sepal length'].mean()))
sepal_width = st.slider("Sepal Width",
                        float(df['sepal width'].min()),
                        float(df['sepal width'].max()),
                        float(df['sepal width'].mean()))
petal_length = st.slider("Petal Length",
                         float(df['petal length'].min()),
                         float(df['petal length'].max()),
                         float(df['petal length'].mean()))
petal_width = st.slider("Petal Width",
                        float(df['petal width'].min()),
                        float(df['petal width'].max()),
                        float(df['petal width'].mean()))

if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    y_pred = clf.predict(input_data)
    predicted_class_name = iris.target_names[y_pred[0]]

    separation()
    st.markdown(
        f"Predicted Iris Flower Type:<p style='font-family:Lucida Console;"
        f"text-align:center;"
        f"margin-top: -20px;"
        f"margin-left: 0%;"
        f"font-size:30px;"
        f"color:cyan;'>{predicted_class_name}</p>",
        unsafe_allow_html=True)

    cm = confusion_matrix(y_test, clf.predict(x_test))  # Fix the order of arguments
    separation(titre2='Metrics', color='violet')
    cm_matrix = pd.DataFrame(
        data=cm,
        columns=[f'Actual:{c}' for c in iris.target_names],
        index=[f'Predicted:{c}' for c in iris.target_names]
    )
    fig = px.imshow(
        cm_matrix,
        labels=dict(x="Actual", y="Predicted", color="Count"),
        color_continuous_scale='greys',
        zmin=0.7,
        title='Confusion Matrix'
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            x=0.8,  # Adjust this value to increase or decrease the distance
        )
    )

    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.plotly_chart(fig)
