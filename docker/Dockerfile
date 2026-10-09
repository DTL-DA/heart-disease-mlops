# La versión de Python y de scikit-learn debe coincidir con la que entrenó
# model.joblib; un modelo serializado con otra versión puede no cargar.
FROM python:3.13-slim
WORKDIR /app
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
