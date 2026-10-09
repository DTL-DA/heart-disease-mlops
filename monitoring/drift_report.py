"""
Etapa 6. Reporte de deriva de datos (data drift) con Evidently.

Compara la distribución de las variables entre los datos de entrenamiento
(referencia) y los datos que recibe el modelo (actuales). Aquí se usa el
conjunto de prueba como datos actuales; en producción se reemplaza por los
registros que llegan a la API.

Uso:  python monitoring/drift_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from src import heart_ml as h  # noqa: E402


def main():
    df = h.cargar_datos()
    X_train, X_test, _, _ = h.dividir(df)
    report = Report(metrics=[DataDriftPreset()])
    resultado = report.run(reference_data=X_train, current_data=X_test)
    salida = h.RAIZ / "drift_report.html"
    resultado.save_html(str(salida))
    print(f"Reporte guardado en {salida}")
    return resultado


if __name__ == "__main__":
    main()
