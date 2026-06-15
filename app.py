from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "df_clean.csv"

st.set_page_config(page_title="Life Expectancy Project", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def build_features(df):
    df_feat = df.copy()
    for col in ["gdp", "population", "measles", "percentage_expenditure"]:
        df_feat[f"log_{col}"] = np.log1p(df_feat[col].clip(lower=0))

    df_feat["vaccine_coverage_avg"] = df_feat[
        ["hepatitis_b", "polio", "diphtheria"]
    ].mean(axis=1)
    df_feat["thinness_avg"] = df_feat[
        ["thinness_119_years", "thinness_59_years"]
    ].mean(axis=1)
    df_feat["status_enc"] = (df_feat["status"] == "Developed").astype(int)
    df_feat["year_norm"] = (
        (df_feat["year"] - df_feat["year"].min())
        / (df_feat["year"].max() - df_feat["year"].min())
    )
    return df_feat


@st.cache_resource
def train_model(df):
    df_feat = build_features(df)
    features = [
        "adult_mortality",
        "log_gdp",
        "schooling",
        "income_composition_of_resources",
        "vaccine_coverage_avg",
        "hivaids",
        "thinness_avg",
        "status_enc",
        "year_norm",
        "total_expenditure",
        "bmi",
    ]
    X = df_feat[features]
    y = df_feat["life_expectancy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred),
    }
    return model, features, metrics


def make_input(features, df):
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())

    st.sidebar.header("Cenario")
    adult_mortality = st.sidebar.slider("Mortalidade adulta", 50, 350, 145)
    gdp = st.sidebar.slider("PIB per capita", 500, 60000, 4200, step=100)
    schooling = st.sidebar.slider("Escolaridade media", 4.0, 18.0, 11.5, step=0.5)
    income = st.sidebar.slider("Composicao de renda", 0.20, 0.95, 0.62, step=0.01)
    vaccine = st.sidebar.slider("Cobertura vacinal media", 40.0, 100.0, 82.0, step=1.0)
    hivaids = st.sidebar.slider("HIV/AIDS", 0.0, 5.0, 0.4, step=0.1)
    thinness = st.sidebar.slider("Magreza media", 0.0, 20.0, 6.5, step=0.5)
    developed = st.sidebar.selectbox("Status", ["Developing", "Developed"])
    year = st.sidebar.slider("Ano de previsao", 2026, 2030, 2027)
    expenditure = st.sidebar.slider("Gasto total em saude", 2.0, 12.0, 6.1, step=0.1)
    bmi = st.sidebar.slider("IMC medio", 15.0, 35.0, 24.5, step=0.5)

    row = pd.DataFrame(
        [
            {
                "adult_mortality": adult_mortality,
                "log_gdp": np.log1p(gdp),
                "schooling": schooling,
                "income_composition_of_resources": income,
                "vaccine_coverage_avg": vaccine,
                "hivaids": hivaids,
                "thinness_avg": thinness,
                "status_enc": int(developed == "Developed"),
                "year_norm": (year - year_min) / (year_max - year_min),
                "total_expenditure": expenditure,
                "bmi": bmi,
            }
        ]
    )
    return row[features]


df = load_data()
model, features, metrics = train_model(df)
scenario = make_input(features, df)
prediction = model.predict(scenario)[0]

st.title("Expectativa de Vida Global")
st.write(
    "Dashboard simples para explorar o modelo Random Forest treinado no projeto."
)

prediction_col, country_col, records_col, r2_col, rmse_col = st.columns(
    [1.7, 1, 1, 1, 1]
)

with prediction_col:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 18px 20px;
            background: #f7f8fa;
        ">
            <div style="
                color: #687487;
                font-size: 0.92rem;
                margin-bottom: 6px;
            ">
                Previsao do cenario
            </div>
            <div style="
                color: #18212f;
                font-size: 2.15rem;
                font-weight: 700;
                line-height: 1.1;
            ">
                {prediction:.2f} anos
            </div>
            <div style="
                color: #687487;
                font-size: 0.82rem;
                margin-top: 6px;
            ">
                Expectativa de vida prevista
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

country_col.metric("Paises", f"{df['country'].nunique():,}")
records_col.metric("Registros", f"{len(df):,}")
r2_col.metric("R2 teste", f"{metrics['R2']:.3f}")
rmse_col.metric("RMSE", f"{metrics['RMSE']:.2f} anos")

st.subheader("Importancia das variaveis")
importance = (
    pd.Series(model.feature_importances_, index=features)
    .sort_values(ascending=False)
    .rename("importance")
    .to_frame()
)
st.bar_chart(importance)
