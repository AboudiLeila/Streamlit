import io
import requests
import streamlit as st
# from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Set page configuration
icon_img = '🦠'
st.set_page_config(
    page_title='COVID-19 Predictions',
    page_icon=icon_img,
    layout="wide",
    initial_sidebar_state='collapsed'
)


def separation(sep1=False , sep2=False, titre1=None, titre2=None, color1='cyan', color2='cyan', lvl1='h1', lvl2 = 'h4', text_align1='center', text_align2='center'):
    if sep1:
        st.text('')
    if titre1:
        st.markdown(
            f"<{lvl1} style='font-family: Lucida Console;text-align: {text_align1}; color: {color1};'>{titre1}</{lvl1}>",
            unsafe_allow_html=True
        )
    if titre2:
        st.markdown(
            f"<{lvl2} style='font-family: Lucida Console;text-align: {text_align2}; color: {color2};'>{titre2}</{lvl2}>",
            unsafe_allow_html=True
        )
    if sep2:
        st.text('')

separation(titre1='COVID-19 Predictions', lvl1='h2', color1='green')

# Fetch the data
api_key = "bfc05aeddcf24111a64128d9e7b2e349"
url = f"https://api.covidactnow.org/v2/states.timeseries.csv?apiKey={api_key}"
response = requests.get(url)
df = pd.read_csv(io.BytesIO(response.content))

# Data preprocessing
df.sample(frac=0.3).sort_index()

df= df[[
    'state',
    'actuals.cases',
    'actuals.vaccinationsCompleted',
    'actuals.deaths'
]]

missing_values = df.isnull().sum()
sorted_columns = missing_values.sort_values()
print(sorted_columns)

df.dropna(inplace=True)
df.isnull().sum()

label_encoder = LabelEncoder()
df[['state_encoded', 'cases_encoded', 'vaccinationsCompleted_encoded', 'deaths_encoded']] = df[['state', 'actuals.cases', 'actuals.vaccinationsCompleted', 'actuals.deaths']].apply(label_encoder.fit_transform)
df.drop(columns=['actuals.deaths'], inplace=True)

data_state=df[['state','state_encoded']]
data_cases=df[['actuals.cases','cases_encoded']]
data_vaccin=df[['actuals.vaccinationsCompleted','vaccinationsCompleted_encoded']]
data_state = data_state.drop_duplicates().reset_index(drop=True)
data_cases = data_cases.drop_duplicates().reset_index(drop=True)
data_vaccin = data_vaccin.drop_duplicates().reset_index(drop=True)

numerical_cols = [
    'state_encoded',
    'cases_encoded',
    'vaccinationsCompleted_encoded',
    'deaths_encoded'
    ]
numerical_df = df[numerical_cols]

scaler = MinMaxScaler()
numerical_df_scaled = scaler.fit_transform(numerical_df)

df_scaled = pd.DataFrame(numerical_df_scaled, columns=numerical_cols)

non_numerical_cols = df.drop(columns=numerical_cols)
df = pd.concat([non_numerical_cols, df_scaled], axis=1)

new_column_names = {
    'state_encoded': 'State',
    'cases_encoded': 'Total_Cases',
    'vaccinationsCompleted_encoded': 'Vaccinations_Completed',
    'deaths_encoded': 'Total_Deaths'
}

df.rename(columns=new_column_names, inplace=True)

df.drop(['state', 'actuals.cases','actuals.vaccinationsCompleted'], axis=1, inplace= True)
df.dropna(axis=0, inplace= True)

separation(titre1='Dataset Sample Scaled',color1='#2464c9', lvl1='h4', text_align1='left')
center_style = """
        <style>
        .center {
            display: flex;
            justify-content: center;
        }
        </style>
    """
st.write(center_style, unsafe_allow_html=True)
st.write("<div class='center'>" + df.sample(frac=0.3).sort_index().head(5).to_html() + "</div>", unsafe_allow_html=True)

separation(titre1='Descriptions',color1='#2464c9', lvl1='h4', text_align1='left')
st.write("<div class='center'>" + df.describe().to_html() + "</div>", unsafe_allow_html=True)


def setFigCenter(seuil1=1, seuil2=4, seuil3=1, cont1=None, figure=None, cont3=None):
    col1, col2, col3 = st.columns([seuil1, seuil2, seuil3])
    with col1:
        st.write(cont1)
    with col2:
        # Save the figure to a BytesIO buffer
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        buffer.seek(0)
        st.image(buffer, use_container_width=True)  # Display the image
    with col3:
        st.write(cont3)

# Histograms with KDE


def plot_histograms_kde(df):
    plt.figure(figsize=(9, 5))
    plt.suptitle('Histograms with KDE of Numerical Columns', fontsize=10)
    for i, col in enumerate(df.columns[:], 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], bins=30, kde=True, color='blue', edgecolor='k')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    return plt.gcf()
separation(titre1='Histograms with KDE of Numerical Columns', color1='#2464c9', lvl1='h4', text_align1='left', sep1=True)
fig_hist_kde = plot_histograms_kde(df)
fig=st.pyplot(fig_hist_kde)
setFigCenter(figure=fig, seuil1=1, seuil2=2, seuil3=1)

# Pairwise Scatter Plots


def plot_pairwise_scatter(df):
    plt.figure(figsize=(10, 7))
    plt.suptitle('Pairwise Scatter Plots of Numerical Columns', y=1.02, fontsize=16)
    g = sns.pairplot(df.iloc[:], diag_kind='kde', markers='o')
    plt.tight_layout()
    return g.fig
separation(titre1='Pairwise Scatter Plots of Numerical Columns', color1='#2464c9', lvl1='h4', text_align1='left', sep1=True)
fig_pairwise_scatter = plot_pairwise_scatter(df)
fig = st.pyplot(fig_pairwise_scatter)
setFigCenter(figure=fig, seuil2=2)

# Plot correlation map


def plot_correlation_map(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.diverging_palette(240, 9, as_cmap=True)
    sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': 0.9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 11}
    )
    plt.xticks(rotation=20)
    plt.yticks(rotation=20)
    return fig

st.set_option('deprecation.showPyplotGlobalUse', False)
separation(titre1='Correlation Map', color1='#2464c9', lvl1='h4', text_align1='left', sep1=True)
fig_correlation_map = plot_correlation_map(df)

# Display the correlation map figure using st.pyplot
st.pyplot(fig_correlation_map)


st.markdown(f"<p style='"
            f"font-size: 14px;"
            f"color: red;"
            f"font-family: italic;"
            f"margin-top: 10px;"
            f"'>The model is running. This can take few moments (~1 to 2 mins)</p>", unsafe_allow_html=True)

# Model training and evaluation
X = df.drop(columns=['Total_Deaths'])
y = df['Total_Deaths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

kf = KFold(n_splits=10, shuffle=False)

param_distributions = {
    'n_estimators': np.arange(200, 500, 100),
    'learning_rate': [0.1, 0.15, 0.2, 0.25],
    'max_depth': [8, 9, 10, 11]
}

xgb_model = xgb.XGBRegressor(random_state=42)

kf=KFold(n_splits=5, shuffle=False)
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions,
    n_iter=10,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1)

random_search.fit(X_train, y_train)

best_xgb_model = random_search.best_estimator_

y_pred = best_xgb_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)


def main():
    # Model training and evaluation
    separation(titre2='Model Training and Evaluation', color1='#2464c9', sep2=True)
    separation(titre1='Best Hyperparameters', color1='#248dc9', lvl1='h6', text_align1='left',
               titre2=random_search.best_params_, color2='red', lvl2='p', text_align2='left', sep2=True)
    separation(titre1='Root Mean Squared Error on Test Set', color1='#248dc9', lvl1='h6', text_align1='left',
               titre2=round(rmse, 3), color2='red', lvl2='p', text_align2='left', sep2=True)

    # Prediction

    separation(titre2='Predictions', color1='#2464c9', sep1=True, sep2=True)
    setFigCenter(seuil1=1,seuil2=1,seuil3=1,cont1=data_state, figure=data_cases, cont3 =data_vaccin)
    features = st.text_area("Enter the features for prediction (comma-separated values)", "State, Total_Cases, Vaccinations_Completed")
    features_list = features.split(',')

    # Check if the first element is 'State' and remove it from the list if it is
    if features_list[0].strip().lower() == 'state':
        features_list = features_list[1:]

    numerical_values = []
    for x in features_list:
        try:
            numerical_values.append(float(x.strip()))
        except ValueError:
            # If the value cannot be converted to float, ignore it
            pass

    if len(numerical_values) != 3:
        st.write("Please enter three numerical values (excluding 'State').")
        return

    input_features = np.array(numerical_values).reshape(1, -1)

    prediction = best_xgb_model.predict(input_features)
    st.write(f"Predicted Total Deaths: {prediction[0]}")

if __name__ == '__main__':
    main()
