import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Tamaño de las imágenes para estandarizar
IMAGE_SIZE = (64, 64) 
BATCH_SIZE = 16 # Ajuste de hiperparámetro (Experimento B)

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int):
    # 1. Crear Datasets
    # La clase ImageFolder encuentra automáticamente las clases por el nombre de las carpetas.
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # 2. Crear DataLoaders
    # Permiten cargar los datos en 'batches' para el entrenamiento.
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, train_data.classes

# --- Configuración de las Transformaciones ---

# EXPERIMENTO B: Transformaciones con Data Augmentation
def get_train_transform_improved(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        # Técnicas de Data Augmentation
        transforms.TrivialAugmentWide(num_magnitude_bins=31), # Gira, voltea, cambia un poco el color
        transforms.ToTensor(), # Convierte la imagen a un tensor de PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# La transformación para la prueba (validación) debe ser SÓLO para redimensionar y normalizar
def get_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])