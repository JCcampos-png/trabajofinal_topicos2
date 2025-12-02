# Actua como el orquestador principal del pipeline. Ejecuta el proceso de ML de principio a fin, invoca los metodos de DataPreprocessor y ModelTrainer en secuencia
# Demostrando un diseño modular, lo que facilita el mantenimiento y la extension futura del proyecto # Se incorporan 4 modelos para la comparacion

#Regresion Logistica, Modelo lineal, alta interpretabilidad
#KNN, modelo basado en distancia, sensible al escalado
#Arbol de Desicion, Modelo no lineal, buena interpretabilidad y baja complejidad
#Random Forest, ensamblaje robusto, mejor rendimiento

#%%
import os
import pandas as pd
from src.module_data import DataPreprocessor
from src.module_ml import ModelTrainer


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# %%

def run_pipeline():
    """Función principal que orquesta el pipeline completo."""
    
    print("--- 🧠 Iniciando Pipeline de Machine Learning WIDS 2024 ---")
    
    # --- 1. PREPARACIÓN DE DATOS ---
    data_proc = DataPreprocessor(data_path='data')
    df_train = data_proc.load_data()
    
    if df_train is None:
        return

    # Separación de objetivo y división de datos
    X, y = data_proc.separate_target(df_train)
    X_train, X_val, y_train, y_val = data_proc.split_data(X, y)
    
    # Creación y aplicación del pipeline de preprocesamiento
    data_proc.create_preprocessing_pipeline(X)
    X_train_proc, X_val_proc, y_train, y_val = data_proc.apply_preprocessing(X_train, X_val, y_train, y_val)

    # --- 2. EXPERIMENTACIÓN Y TRACKING (MLflow) ---
    trainer = ModelTrainer(experiment_name="WIDS_Datathon_2024_Comparacion_Modelos")
    
    # Definición de los modelos a comparar con hiperparámetros sencillos
    models = {
        "RegresionLogistica": LogisticRegression(solver='liblinear', random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42), # Profundidad limitada para evitar overfitting y mantener interpretabilidad
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "KNN_Classifier": KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    
    # Bucle para entrenar y evaluar cada modelo
    for name, model in models.items():
        acc, auc, time = trainer.train_and_evaluate(
            model=model,
            model_name=name,
            X_train=X_train_proc,
            X_val=X_val_proc,
            y_train=y_train,
            y_val=y_val
        )
        results.append({
            'Modelo': name,
            'ROC-AUC': round(auc, 4),
            'Accuracy': round(acc, 4),
            'Tiempo (s)': round(time, 2),
        })
        
    # --- 3. COMPARACIÓN DE RESULTADOS ---
    print("\n\n--- Resumen de Comparación de Modelos (Ordenado por ROC-AUC) ---")
    results_df = pd.DataFrame(results)
    
    # Agregar análisis de interpretabilidad y complejidad para el informe
    results_df['Interpretabilidad'] = results_df['Modelo'].apply(
        lambda x: 'Alta' if x in ['RegresionLogistica', 'DecisionTree'] else 'Media/Baja'
    )
    
    results_df = results_df.sort_values(by='ROC-AUC', ascending=False)
    
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    # La carpeta 'data' es necesaria para que el módulo de datos funcione
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Carpeta 'data/' creada. Por favor, coloque los archivos CSV dentro.")
    
    # Ejecutar el pipeline
    run_pipeline()

# --- Fin de main.py ---
# %%
