# Heart Disease MLOps

Proyecto Integrador de Aprendizaje Automático · Donnys Torres Lozano · Maestría en Analítica de Datos, Universidad del Norte

Flujo completo de MLOps en entorno local para predecir enfermedad cardíaca con el dataset *Heart Failure Prediction* (`heart.csv`, 918 pacientes). El proyecto va desde el análisis de los datos hasta el monitoreo del modelo: preprocesamiento sin fuga de datos, comparación de modelos con `Pipeline` y `GridSearchCV`, API con FastAPI, contenedor Docker, despliegue en Kubernetes, integración continua con GitHub Actions y reporte de deriva con Evidently.

## Resultados

Se compararon siete clasificadores con validación cruzada estratificada de 5 pliegues. El modelo se elige por el AUC de validación cruzada, nunca por el conjunto de prueba.

| Puesto | Modelo | AUC validación cruzada | AUC prueba | Exactitud prueba |
|---|---|---|---|---|
| 1 | RandomForest | 0,931 ± 0,018 | 0,931 | 0,880 |
| 2 | GradientBoosting | 0,929 ± 0,009 | 0,924 | 0,886 |
| 3 | LogisticRegression | 0,918 ± 0,024 | 0,935 | 0,880 |
| 4 | SVC | 0,917 ± 0,037 | 0,931 | 0,853 |
| 5 | KNN | 0,916 ± 0,029 | 0,928 | 0,870 |
| 6 | GaussianNB | 0,908 ± 0,015 | 0,909 | 0,875 |
| 7 | DecisionTree | 0,905 ± 0,026 | 0,886 | 0,842 |

El modelo desplegado es un RandomForest (400 árboles, profundidad máxima 8). En prueba clasifica bien a 162 de 184 pacientes, con 10 falsos negativos y 12 falsos positivos. Los cinco primeros modelos son estadísticamente equivalentes: sus diferencias son menores que la desviación entre pliegues.

## Estructura

```
heart-disease-mlops/
├── app/
│   ├── api.py                  # API FastAPI
│   ├── model.joblib            # Pipeline entrenado (preprocesamiento + modelo)
│   └── model_info.json         # Modelo, hiperparámetros y métricas
├── data/heart.csv
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb   # Etapa 1: EDA, fuga de datos y ranking de modelos
│   └── 2_model_pipeline_cv.ipynb    # Etapa 2: Pipeline + GridSearchCV, matriz de confusión, ROC
├── src/heart_ml.py             # Funciones reutilizables de carga, entrenamiento y evaluación
├── monitoring/drift_report.py  # Etapa 6: genera drift_report.html
├── tests/test_api.py           # Pruebas automáticas de la API
├── .github/workflows/ci.yml    # Etapa 5: lint, pruebas y construcción de la imagen
├── train.py                    # Entrena, compara y exporta el modelo
├── drift_report.html
├── model.joblib
├── ranking_modelos.csv
└── README.md
```

## Cómo ejecutar cada etapa

### Etapas 1 y 2. Entrenamiento

```bash
pip install -r requirements-dev.txt
python train.py                      # compara los 7 modelos y exporta app/model.joblib
jupyter notebook notebooks/          # cuadernos con el análisis y las interpretaciones
```

### Etapa 3. API con FastAPI y Docker

Sin Docker:

```bash
uvicorn app.api:app --port 8000
```

Con Docker:

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

La documentación interactiva queda en http://localhost:8000/docs. Ejemplo de predicción:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Age": 58, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 260,
  "FastingBS": 1, "RestingECG": "Normal", "MaxHR": 115, "ExerciseAngina": "Y",
  "Oldpeak": 2.0, "ST_Slope": "Flat"}'
```

Respuesta:

```json
{"heart_disease_probability": 0.9767, "prediction": 1}
```

| Ruta | Método | Qué devuelve |
|---|---|---|
| `/predict` | POST | Probabilidad de enfermedad y predicción (1 si la probabilidad supera 0,5) |
| `/health` | GET | Estado del servicio; lo usan las sondas de Kubernetes |
| `/model-info` | GET | Modelo desplegado, hiperparámetros y métricas |
| `/docs` | GET | Documentación interactiva |

### Etapa 4. Kubernetes local con Minikube

```bash
minikube start
minikube image load heart-api:latest       # usa la imagen local, sin Docker Hub
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
minikube service heart-service --url       # dirección para probar la API
```

Para usar una imagen publicada en Docker Hub, cambie `image` en `k8s/deployment.yaml` por `<TU_USUARIO_DOCKER>/heart-api` y elimine la línea `imagePullPolicy: Never`.

### Etapa 5. Integración continua

El flujo `.github/workflows/ci.yml` se ejecuta en cada `push`: instala dependencias, revisa el estilo con `flake8`, corre las pruebas con `pytest` y construye la imagen Docker. Localmente:

```bash
flake8 app/ src/ tests/ train.py
pytest tests/ -v
```

### Etapa 6. Monitoreo de deriva de datos

```bash
python monitoring/drift_report.py          # genera drift_report.html
```

El reporte compara entrenamiento (referencia) contra prueba (datos actuales). Ninguna de las 11 variables presenta deriva al 5 %; la más cercana es el colesterol, con un valor p de 0,067. Este resultado es el esperado, porque ambos conjuntos salen de la misma partición estratificada, y sirve como línea de referencia. En producción, los datos actuales se reemplazan por los registros que recibe la API.

## Decisiones y ajustes frente al enunciado

| Enunciado | Proyecto | Razón |
|---|---|---|
| Columna objetivo `target` | `HeartDisease` | Es el nombre real en `heart.csv` |
| `MinMaxScaler` sobre todo `X` | `ColumnTransformer` dentro del `Pipeline` | Hay 5 columnas de texto que deben codificarse antes de escalar |
| Sin tratamiento de faltantes | Colesterol = 0 imputado con la mediana | 172 registros traen el colesterol en cero, y no falta al azar: 88,4 % de ellos tiene enfermedad |
| `features: list` en la API | Campos con nombre y validación | El modelo necesita los nombres de columna, y la validación rechaza datos inválidos |
| `python:3.10-slim` | `python:3.13-slim` | scikit-learn 1.9.1, con el que se entrenó el modelo, requiere una versión más reciente de Python |
| `evidently.report` | `from evidently import Report` | La interfaz cambió en Evidently 0.7 |
| El flujo «sin fuga» del ejemplo conserva `leaky_feature` | Los tipos de fuga se demuestran por separado | En el ejemplo original ambos AUC salen cercanos a 1 y la comparación no muestra diferencia |

## Limitaciones

La muestra es de pacientes remitidos por sospecha clínica, con 55,3 % de enfermos, de modo que la precisión sería menor en población general. El 79 % son hombres y el desempeño en mujeres no está validado. El dataset combina cinco fuentes sin identificarlas, por lo que no se puede comprobar si el modelo generaliza entre hospitales. El conjunto de prueba tiene 184 pacientes, y las diferencias entre modelos equivalen a uno o dos de ellos. El proyecto es académico y no es una herramienta de diagnóstico.
