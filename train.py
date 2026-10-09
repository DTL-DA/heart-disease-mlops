"""
Entrena todos los modelos con Pipeline + GridSearchCV, elige el de mayor AUC en
validación cruzada y lo exporta para la API.

Uso:  python train.py
"""
import joblib

from src import heart_ml as h


def main():
    df = h.cargar_datos()
    X_train, X_test, y_train, y_test = h.dividir(df)
    ranking, ajustados = h.comparar_modelos(X_train, y_train, X_test, y_test)

    # El modelo se elige por validación cruzada, nunca por el conjunto de prueba.
    ganador = ranking.iloc[0]["Modelo"]
    modelo = ajustados[ganador].best_estimator_

    for ruta in (h.RAIZ / "app" / "model.joblib", h.RAIZ / "model.joblib"):
        joblib.dump(modelo, ruta)

    resumen = ranking.drop(columns="Mejores hiperparámetros").round(4)
    resumen.to_csv(h.RAIZ / "ranking_modelos.csv")
    print(resumen.to_string())
    print(f"\nModelo exportado: {ganador} -> app/model.joblib")


if __name__ == "__main__":
    main()
