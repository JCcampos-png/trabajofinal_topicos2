#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

#%%

class DataPreprocessor:
    """Clase para la carga, limpieza, preprocesamiento y división de datos WIDS 2024."""
    
    def __init__(self, data_path='data', target_col='DiagPeriodL90D', test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        
    def load_data(self, filename='training.csv'):
        """Carga el conjunto de datos de entrenamiento desde la carpeta 'data/'."""
        file_path = os.path.join(self.data_path, filename)
        try:
            df = pd.read_csv(file_path)
            print(f"Datos cargados. Dimensiones: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: Archivo {file_path} no encontrado.")
            return None
    
    def separate_target(self, df):
        """Separa las características (X) y la variable objetivo (y)."""
        if df is None or self.target_col not in df.columns:
            return None, None
        
        # Columnas a eliminar según el EDA: IDs y variables con >99% de nulos
        cols_to_drop = [
            'patient_id', 
            'metastatic_first_novel_treatment', 
            'metastatic_first_novel_treatment_type'
        ]
        
        X = df.drop([self.target_col] + cols_to_drop, axis=1, errors='ignore')
        y = df[self.target_col]
        return X, y

    def create_preprocessing_pipeline(self, X):
        """Define el pipeline de preprocesamiento usando ColumnTransformer."""
        
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        
        # 1. Transformer para variables Numéricas: Imputación con mediana y escalado
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ])
        
        # 2. Transformer para variables Categóricas: Imputación con 'Missing' y One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # ColumnTransformer combina las transformaciones
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' # Descartar otras columnas
        )
        print("Pipeline de preprocesamiento creado: Imputación y Escalado/Codificación.")

    def split_data(self, X, y):
        """Divide los datos en conjuntos de entrenamiento (Train) y validación (Val)."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"Datos divididos. Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")
        return X_train, X_val, y_train, y_val

    def apply_preprocessing(self, X_train, X_val, y_train, y_val):
        """Aplica el ajuste y la transformación a los conjuntos de datos."""
        
        if self.preprocessor is None:
            raise Exception("Debe llamar a create_preprocessing_pipeline primero.")
        
        # Ajustar (fit) el transformador SÓLO en el conjunto de entrenamiento
        self.preprocessor.fit(X_train)
        
        # Aplicar la transformación (transform) a ambos conjuntos
        X_train_processed = self.preprocessor.transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        
        print(f"Datos preprocesados. Features resultantes: {X_train_processed.shape[1]}")
        return X_train_processed, X_val_processed, y_train, y_val
    
    # fin del module_data.py
    
# %%
