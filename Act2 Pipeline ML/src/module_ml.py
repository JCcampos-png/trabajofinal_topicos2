# Modulo para entrenar, evaluar y registrar los experimentos usando MLflow
# se implementa la experimentacion. Ingregracion con MLflow que permite el tracking de cada corrido de modelo
# se una el contexto with mlflow.start_run() as run: para registrar automaticamente:
#Parametros: Nombre del modelo y sus hiperparámetros.
# Metricas: accuracy y roc_auc (métrica clave para clasificación desbalanceada).
#Artefactos: El modelo entrenado, listo para ser desplegado.
#Tiempos: El tiempo de cómputo, útil para la comparación de eficiencia.

#%%

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import time
# %%

class ModelTrainer:
    """Clase para el entrenamiento, evaluación y tracking de modelos ML con MLflow."""
    
    def __init__(self, experiment_name="WIDS_Datathon_2024_Pipeline"):
        # Configuramos MLflow. Si el experimento no existe, lo crea.
        mlflow.set_experiment(experiment_name)
        print(f"MLflow configurado. Experimento: '{experiment_name}'")

    def train_and_evaluate(self, model, model_name, X_train, X_val, y_train, y_val):
        """Entrena un modelo, evalúa con métricas clave y registra el experimento."""
        
        print(f"\n--- Iniciando entrenamiento de {model_name} ---")
        start_time = time.time()
        
        # Iniciar el ciclo de tracking con MLflow
        with mlflow.start_run() as run:
            
            # 1. Registro de Parámetros
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", len(y_val) / (len(y_train) + len(y_val)))
            mlflow.log_params(model.get_params()) # Registra hiperparámetros del modelo
            
            # 2. Entrenamiento
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 3. Predicciones y Evaluación
            y_pred = model.predict(X_val)
            # Obtenemos las probabilidades para el cálculo del ROC-AUC
            y_proba = model.predict_proba(X_val)[:, 1] 
            
            # Cálculo de Métricas
            accuracy = accuracy_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)
            
            # 4. Registro de Métricas y Tiempos
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("training_time_sec", training_time)

            print(f"  Accuracy (Validación): {accuracy:.4f}")
            print(f"  ROC-AUC (Validación): {roc_auc:.4f}")
            print(f"  Tiempo de Entrenamiento: {training_time:.2f} segundos")
            
            # 5. Registro del Modelo como Artefacto
            mlflow.sklearn.log_model(model, "model")
            
            # Opcional: Imprimir reporte de clasificación
            print("\n  Reporte de Clasificación:")
            print(classification_report(y_val, y_pred))

        return accuracy, roc_auc, training_time

# --- Fin de module_ml.py ---
# %%
