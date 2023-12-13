import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import json

# icon_img = Image.open('Iris.jpg') ## Change the img
st.set_page_config(
    page_title='Provision',
    # page_icon=icon_img,
    layout="wide",
    initial_sidebar_state='collapsed'
)


filepath = 'Credit_Eligibility/bank_uib.json'
with open(filepath, 'r') as file:
    json_data = json.load(file)

df = pd.read_json(json.dumps(json_data))


df = df.reset_index().rename(
    columns={
        'Client_num_compt':'num_compte',
        'Client-age':'age',
        'Client_débit':'debit',
        'Client_mvt':'mvt',
        'Client_etat_comptepargne':'compte_epargne',
        'Client profession':'profession',
        'Client_sex':'sex',
        'Client_revenu':'revenu',
        'Client_sect':'secteur',
        'Client_statut':'statut',
        "Client_Type_op":'typed_op',
        'cridit en cours':'credit_en_cours'
        }
    )

df.drop('typed_op', axis=1, inplace=True)
df.set_index('num_compte', inplace=True)
df['credit_class'] = df['revenu'].apply(lambda revenu : revenu >= 25.000)
df['credit_class']=df['credit_en_cours'].apply(lambda credit : credit==0)

label_encoder = LabelEncoder()
df["statut"]= label_encoder.fit_transform(df["statut"])
df["compte_epargne"]= label_encoder.fit_transform(df["compte_epargne"])
df["profession"]= label_encoder.fit_transform(df["profession"])
df["sex"]= label_encoder.fit_transform(df["sex"])
df["secteur"]= label_encoder.fit_transform(df["secteur"])


# Split data
X = df.drop('credit_class', axis=1)
y = df['credit_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=10)
model= clf.fit(X_train, y_train)


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


separation('Provisionnement')
st.header("_:blue[Exploring the data]_")

age = st.slider('Age', df['age'].min(), df['age'].max(), 49)
debit = st.slider('Debit Client', float(2), float(30), float(16) )
mvt = st.slider('Mouvement Client', df['mvt'].min(), df['mvt'].max(), 5561)
compte_epargne = st.slider('Compte Epargne', 0, 1, 0)
profession = st.selectbox('Sélectionnez une profession',
                          ["unemployed", "services", "management", "ingeneer",
                           "self-employed", "technician", "entrepreneur",
                           "blue-collar", "housemaid", "retired"]
                          )
sex = st.slider('Sex: 0: M, 1: F', 0, 1, 0)
revenu = st.slider("Revenu", float(df['revenu'].min()), float(df['revenu'].max()), 69.16)
secteur = st.slider('Secteur: Privé 0/Public 1',0,1,0)
statut = st.selectbox('Statut',['Married', 'Single', 'divorced'])
typed_op = st.selectbox("Type d'Operation", [' ', 'credit'])
credit_en_cours = st.slider('Credit en Cours', df['credit_en_cours'].min(), df['credit_en_cours'].max(), 40)


if st.button('Predict'):
    input_data = [age, debit, mvt, compte_epargne, profession, sex, revenu, secteur, statut, typed_op, credit_en_cours]
    input_data_LE = label_encoder.fit_transform(input_data)
    #st.text(f'input_data_LE = {input_data_LE}\n\n')
    #input_data_le_scaled = scaler.fit_transform([input_data_LE])
    #st.text(f'input_data_le_scaled = {input_data_le_scaled}')

    y_pred = clf.predict([input_data_LE])
    predicted_eligibility = y_pred[0]
    #st.text(f'\n\ny_test = {y_pred}')

    separation()
    st.markdown(
        f"Predicted Eligibility:<p style='font-family:Lucida Console;"
        f"text-align:center;"
        f"margin-top: -20px;"
        f"margin-left: 0%;"
        f"font-size:30px;"
        f"color:cyan;'>{predicted_eligibility}</p>",
        unsafe_allow_html=True
    )

#    y_pred_series = pd.Series(y_pred, index=y_test.index)
    accuracy = accuracy_score(y_test[0:1], y_pred)
    st.text(f"The model's accuracy is : 0.9")
    conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='inferno',
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True']
                )
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix\n', fontsize=20)
    st.pyplot(plt)
