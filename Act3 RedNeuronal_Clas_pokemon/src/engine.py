import torch
from typing import Tuple, Dict, List

# Función auxiliar para calcular la precisión
def accuracy_fn(y_true, y_pred):
    """Calcula la precisión entre las etiquetas verdaderas y las predicciones."""
    # Compara la predicción con la etiqueta verdadera
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# ---------------------------------------------------------------- #

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> Tuple[float, float]:
    """
    Realiza un paso de entrenamiento sobre todos los batches del DataLoader.
    Devuelve la pérdida promedio y la precisión promedio del entrenamiento.
    """
    model.train() # Pone el modelo en modo entrenamiento (ej. activa Dropout)
    
    train_loss, train_acc = 0, 0
    
    # Recorre todos los batches del DataLoader
    for batch, (X, y) in enumerate(dataloader):
        # Mueve datos al dispositivo objetivo (CPU o GPU)
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass (Predicción)
        y_pred_logits = model(X) # Obtiene los 'logits' (puntuaciones antes de softmax)
        
        # 2. Calcular la pérdida (Loss)
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()

        # 3. Optimizar (Paso Clave para el Aprendizaje)
        optimizer.zero_grad() # Pone los gradientes a cero antes de cada backpropagation
        loss.backward()       # Backpropagation: calcula los gradientes de la pérdida respecto a los parámetros
        optimizer.step()      # Ajusta los parámetros del modelo usando el optimizador
        
        # 4. Calcular la precisión
        y_pred_class = torch.argmax(y_pred_logits, dim=1) # Convierte logits a la clase predicha (índice)
        train_acc += accuracy_fn(y, y_pred_class)

    # Devuelve la pérdida y precisión promedio por época
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# ---------------------------------------------------------------- #

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: torch.device) -> Tuple[float, float]:
    """
    Realiza un paso de prueba/validación sobre todos los batches del DataLoader.
    Devuelve la pérdida promedio y la precisión promedio de la prueba.
    """
    model.eval() # Pone el modelo en modo evaluación (desactiva Dropout, Batchnorm, etc.)
    
    test_loss, test_acc = 0, 0
    
    # Desactiva el cálculo de gradientes; no es necesario y ahorra memoria/tiempo
    with torch.inference_mode():
        # Recorre todos los batches del DataLoader
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass (Predicción)
            y_pred_logits = model(X)
            
            # 2. Calcular la pérdida
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()
            
            # 3. Calcular la precisión
            y_pred_class = torch.argmax(y_pred_logits, dim=1)
            test_acc += accuracy_fn(y, y_pred_class)

    # Devuelve la pérdida y precisión promedio por época
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# ---------------------------------------------------------------- #

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """
    Función principal que entrena y prueba el modelo por un número de épocas.
    Registra las métricas de entrenamiento y validación.
    """
    # 1. Crear diccionario para guardar los resultados del entrenamiento
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    # 2. Bucle principal de épocas
    for epoch in range(epochs):
        # A. Paso de Entrenamiento
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        # B. Paso de Prueba/Validación
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # 3. Imprimir y guardar resultados
        print(
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

        # Guardar resultados en el diccionario para la gráfica final
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 4. Devolver los resultados completos
    return results