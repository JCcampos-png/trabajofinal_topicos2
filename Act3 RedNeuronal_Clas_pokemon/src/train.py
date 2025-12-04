import os
import torch
from pathlib import Path
from src import data_setup, model_builder, engine

# Directorios de datos
DATA_PATH = Path("data")
TRAIN_DIR = DATA_PATH / "train"
TEST_DIR = DATA_PATH / "test"

# Hiperparámetros
NUM_EPOCHS = 20
HIDDEN_UNITS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 16 # Ajuste para el Experimento B

# 1. Configuración del Dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Preparación de Datos y DataLoaders (Experimento B Mejorado)
train_transform = data_setup.get_train_transform_improved(data_setup.IMAGE_SIZE)
test_transform = data_setup.get_test_transform(data_setup.IMAGE_SIZE)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(TRAIN_DIR),
    test_dir=str(TEST_DIR),
    transform=train_transform,
    batch_size=BATCH_SIZE
)

# 3. Construcción del Modelo (Modelo Mejorado con 3 bloques)
model = model_builder.PokemonVGG_Improved(
    input_shape=3, # Las imágenes RGB tienen 3 canales
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names) # 151 Pokémon
).to(device)

# 4. Definición de Pérdida y Optimizador
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Entrenamiento
# La función 'train' viene de engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=NUM_EPOCHS,
             device=device)

# 6. Guardar el modelo entrenado
# Se guarda el modelo para usarlo después.
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / "pokemon_vgg_improved.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)