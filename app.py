from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "df_clean.csv"
SCENARIO_DEFAULTS = {
    "adult_mortality": 145,
    "gdp": 4200,
    "schooling": 11.5,
    "income": 0.62,
    "vaccine": 82.0,
    "hivaids": 0.4,
    "thinness": 6.5,
    "status": "Developing",
    "year": 2027,
    "expenditure": 6.1,
    "bmi": 24.5,
}
SCENARIO_LIMITS = {
    "adult_mortality": (50, 350),
    "gdp": (500, 60000),
    "schooling": (4.0, 18.0),
    "income": (0.20, 0.95),
    "vaccine": (40.0, 100.0),
    "hivaids": (0.0, 5.0),
    "thinness": (0.0, 20.0),
    "year": (2026, 2030),
    "expenditure": (2.0, 12.0),
    "bmi": (15.0, 35.0),
}
SENSITIVITY_OPTIONS = {
    "Mortalidade adulta": "adult_mortality",
    "PIB per capita": "gdp",
    "Escolaridade media": "schooling",
    "Composicao de renda": "income",
    "Cobertura vacinal media": "vaccine",
    "HIV/AIDS": "hivaids",
    "Magreza media": "thinness",
    "Gasto total em saude": "expenditure",
    "IMC medio": "bmi",
}

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
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, features


def make_feature_row(values, features, df):
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    row = pd.DataFrame(
        [
            {
                "adult_mortality": values["adult_mortality"],
                "log_gdp": np.log1p(values["gdp"]),
                "schooling": values["schooling"],
                "income_composition_of_resources": values["income"],
                "vaccine_coverage_avg": values["vaccine"],
                "hivaids": values["hivaids"],
                "thinness_avg": values["thinness"],
                "status_enc": int(values["status"] == "Developed"),
                "year_norm": (values["year"] - year_min) / (year_max - year_min),
                "total_expenditure": values["expenditure"],
                "bmi": values["bmi"],
            }
        ]
    )
    return row[features]


def make_input(features, df):
    st.sidebar.header("Cenario")
    values = {
        "adult_mortality": st.sidebar.slider("Mortalidade adulta", 50, 350, 145),
        "gdp": st.sidebar.slider("PIB per capita", 500, 60000, 4200, step=100),
        "schooling": st.sidebar.slider("Escolaridade media", 4.0, 18.0, 11.5, step=0.5),
        "income": st.sidebar.slider("Composicao de renda", 0.20, 0.95, 0.62, step=0.01),
        "vaccine": st.sidebar.slider("Cobertura vacinal media", 40.0, 100.0, 82.0, step=1.0),
        "hivaids": st.sidebar.slider("HIV/AIDS", 0.0, 5.0, 0.4, step=0.1),
        "thinness": st.sidebar.slider("Magreza media", 0.0, 20.0, 6.5, step=0.5),
        "status": st.sidebar.selectbox("Status", ["Developing", "Developed"]),
        "year": st.sidebar.slider("Ano de previsao", 2026, 2030, 2027),
        "expenditure": st.sidebar.slider("Gasto total em saude", 2.0, 12.0, 6.1, step=0.1),
        "bmi": st.sidebar.slider("IMC medio", 15.0, 35.0, 24.5, step=0.5),
    }
    return make_feature_row(values, features, df), values


def predict_scenario(model, features, df, values):
    row = make_feature_row(values, features, df)
    return float(model.predict(row)[0])


def clipped_values(values, **updates):
    changed = values.copy()
    for key, value in updates.items():
        if key in SCENARIO_LIMITS:
            low, high = SCENARIO_LIMITS[key]
            value = min(max(value, low), high)
        changed[key] = value
    return changed


def scenario_variants(model, features, df, values):
    variants = {
        "Cenario atual": values,
        "+1 ano de escolaridade": clipped_values(
            values, schooling=values["schooling"] + 1
        ),
        "+5 p.p. de vacina": clipped_values(values, vaccine=values["vaccine"] + 5),
        "-10% mortalidade adulta": clipped_values(
            values, adult_mortality=values["adult_mortality"] * 0.9
        ),
        "Acoes combinadas": clipped_values(
            values,
            schooling=values["schooling"] + 1,
            vaccine=values["vaccine"] + 5,
            income=values["income"] + 0.03,
            adult_mortality=values["adult_mortality"] * 0.9,
        ),
    }
    return pd.DataFrame(
        {
            "cenario": name,
            "expectativa_vida": predict_scenario(model, features, df, item),
        }
        for name, item in variants.items()
    )


def sensitivity_curve(model, features, df, values, option):
    key = SENSITIVITY_OPTIONS[option]
    low, high = SCENARIO_LIMITS[key]
    grid = np.linspace(low, high, 25)
    predictions = []
    for value in grid:
        adjusted = clipped_values(values, **{key: value})
        predictions.append(predict_scenario(model, features, df, adjusted))

    return pd.DataFrame({option: grid, "expectativa_vida": predictions})


def year_projection(model, features, df, values):
    years = range(SCENARIO_LIMITS["year"][0], SCENARIO_LIMITS["year"][1] + 1)
    return pd.DataFrame(
        {
            "ano": year,
            "expectativa_vida": predict_scenario(
                model, features, df, clipped_values(values, year=year)
            ),
        }
        for year in years
    )


df = load_data()
model, features = train_model(df)
scenario, scenario_values = make_input(features, df)
prediction = model.predict(scenario)[0]
base_prediction = predict_scenario(model, features, df, SCENARIO_DEFAULTS)
same_status = df.loc[df["status"] == scenario_values["status"], "life_expectancy"]
status_mean = same_status.mean()
status_percentile = (same_status <= prediction).mean() * 100

st.title("Expectativa de Vida Global")

prediction_col, baseline_col, status_col, percentile_col = st.columns(
    [1.8, 1, 1, 1]
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

baseline_col.metric(
    "Vs configuracao inicial", f"{prediction - base_prediction:+.2f} anos"
)
status_col.metric("Vs media do status", f"{prediction - status_mean:+.2f} anos")
percentile_col.metric("Percentil historico", f"{status_percentile:.0f}%")

left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.subheader("Cenarios alternativos")
    variants_df = scenario_variants(model, features, df, scenario_values)
    st.bar_chart(
        variants_df.set_index("cenario")["expectativa_vida"],
        y_label="Expectativa de vida prevista",
    )

    st.subheader("Projecao por ano")
    projection_df = year_projection(model, features, df, scenario_values)
    st.line_chart(
        projection_df.set_index("ano")["expectativa_vida"],
        y_label="Expectativa de vida prevista",
    )

with right_col:
    st.subheader("Sensibilidade do cenario")
    selected_option = st.selectbox(
        "Variavel testada",
        list(SENSITIVITY_OPTIONS),
        index=2,
    )
    sensitivity_df = sensitivity_curve(
        model, features, df, scenario_values, selected_option
    )
    st.line_chart(
        sensitivity_df.set_index(selected_option)["expectativa_vida"],
        y_label="Expectativa de vida prevista",
    )

    st.subheader("Comparacao historica")
    comparison_df = pd.DataFrame(
        {
            "referencia": [
                "Cenario atual",
                "Media mesmo status",
                "Mediana mesmo status",
                "Top 25% mesmo status",
            ],
            "expectativa_vida": [
                prediction,
                status_mean,
                same_status.median(),
                same_status.quantile(0.75),
            ],
        }
    )
    st.bar_chart(
        comparison_df.set_index("referencia")["expectativa_vida"],
        y_label="Expectativa de vida",
    )
