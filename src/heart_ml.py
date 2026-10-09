"""
Funciones reutilizables del proyecto: carga, limpieza, preprocesamiento,
entrenamiento con GridSearchCV y evaluación.

Las usan los cuadernos, el script train.py y el reporte de monitoreo, de modo
que todos trabajan con exactamente la misma lógica.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "heart.csv"
SEMILLA = 42

OBJETIVO = "HeartDisease"
NUMERICAS = ["Age", "RestingBP", "MaxHR", "Oldpeak"]
COLESTEROL = ["Cholesterol"]
CATEGORICAS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
BINARIAS = ["FastingBS"]
COLUMNAS = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
            "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]


def cargar_datos(ruta=RUTA_DATOS):
    """Lee heart.csv, quita nulos y repetidos y elimina RestingBP = 0 (valor imposible)."""
    df = pd.read_csv(ruta).dropna().drop_duplicates()
    df = df[df["RestingBP"] != 0].reset_index(drop=True)
    return df


def dividir(df, proporcion_prueba=0.20):
    """División estratificada ANTES de cualquier transformación."""
    X, y = df[COLUMNAS], df[OBJETIVO]
    return train_test_split(X, y, test_size=proporcion_prueba, stratify=y, random_state=SEMILLA)


def construir_preprocesador():
    """
    Preprocesamiento que se ajusta solo con entrenamiento, dentro del Pipeline.

    - Cholesterol: el valor 0 es un dato faltante; se imputa con la mediana y se escala.
    - Numéricas: MinMaxScaler.
    - Categóricas: one-hot.
    - FastingBS: ya es 0/1, pasa sin cambio.
    """
    colesterol = Pipeline([("imputar", SimpleImputer(missing_values=0, strategy="median")),
                           ("escalar", MinMaxScaler())])
    return ColumnTransformer([
        ("col", colesterol, COLESTEROL),
        ("num", MinMaxScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("bin", "passthrough", BINARIAS),
    ])


def catalogo_modelos():
    """Modelos y rejillas de hiperparámetros que se comparan."""
    return {
        "SVC": (SVC(probability=True, random_state=SEMILLA),
                {"clf__C": [0.1, 1, 10], "clf__gamma": [0.01, 0.1, "scale"]}),
        "LogisticRegression": (LogisticRegression(max_iter=2000, random_state=SEMILLA),
                               {"clf__C": [0.01, 0.1, 1, 10]}),
        "RandomForest": (RandomForestClassifier(random_state=SEMILLA),
                         {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 8, None],
                          "clf__min_samples_leaf": [1, 5]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=SEMILLA),
                             {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                              "clf__max_depth": [2, 3]}),
        "DecisionTree": (DecisionTreeClassifier(random_state=SEMILLA),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 20]}),
        "GaussianNB": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}),
    }


def train_pipeline(X_train, y_train, model, param_grid, cv=5):
    """Entrena un Pipeline (preprocesamiento + modelo) con GridSearchCV y AUC como criterio."""
    pipe = Pipeline([("prep", construir_preprocesador()), ("clf", model)])
    pliegues = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEMILLA)
    grid = GridSearchCV(pipe, param_grid, cv=pliegues, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def evaluar(modelo, X, y):
    """Métricas de clasificación de un modelo ya entrenado."""
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, proba),
        "Exactitud": accuracy_score(y, pred),
        "Precisión": precision_score(y, pred),
        "Sensibilidad": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "matriz": confusion_matrix(y, pred),
    }


def comparar_modelos(X_train, y_train, X_test, y_test):
    """Entrena todos los modelos del catálogo y devuelve (ranking, rejillas ajustadas)."""
    filas, ajustados = [], {}
    for nombre, (modelo, rejilla) in catalogo_modelos().items():
        grid = train_pipeline(X_train, y_train, modelo, rejilla)
        ajustados[nombre] = grid
        m = evaluar(grid.best_estimator_, X_test, y_test)
        i = grid.best_index_
        filas.append({
            "Modelo": nombre,
            "AUC CV (media)": grid.best_score_,
            "AUC CV (desv.)": grid.cv_results_["std_test_score"][i],
            "AUC prueba": m["AUC"], "Exactitud prueba": m["Exactitud"],
            "Precisión prueba": m["Precisión"], "Sensibilidad prueba": m["Sensibilidad"],
            "Mejores hiperparámetros": {k.replace("clf__", ""): v for k, v in grid.best_params_.items()},
        })
    ranking = (pd.DataFrame(filas).sort_values("AUC CV (media)", ascending=False)
               .reset_index(drop=True))
    ranking.index = ranking.index + 1
    ranking.index.name = "Puesto"
    return ranking, ajustados
