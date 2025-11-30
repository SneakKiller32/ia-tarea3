# Imports necesarios
# Numpy para arreglos numericos
import numpy as np
# tensorflow para construir y entrenar CNNs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore
# sklearn para dividir dataset y métricas
from sklearn.model_selection import train_test_split
# matplotlib para graficar curvas de aprendizaje
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 1. Carga y manejo del dataset
# Las imagenes provienen de carpetas con la clase
def load_and_organize_dataset(data_path):

    """
    Carga las imagenes y las organiza en conjuntos de entrenamiento,
    validacion y prueba.
    
    Organizacion:
    - 70% Entrenamiento
    - 15% Validacion
    - 15% Prueba
    """

    print("\n1. Carga y manejo del dataset\n")
    
    # Listas vacias para guardar datos
    # Almacenar imagenes
    images = [] 
    # Almacenar etiquetas
    labels = [] 
    # Nombres de clases
    class_names = [] 
    
    # Cargar imagenes desde carpetas organizadas por clase
    data_dir = Path(data_path)
    
    # Iterar sobre cada carpeta de clase
    # iterdir() lista todos los archivos y carpetas en data_dir
    # Mientras que enumerate() da un indice (class_idx) y el elemento (class_folder)
    # sorted() asegura un orden
    # class_idx sera la etiqueta numerica
    # y class_folder.name sera el nombre de la clase

    for class_idx, class_folder in enumerate(sorted(data_dir.iterdir())):
        # Verificacion basica de dir
        if class_folder.is_dir():
            # append de los nombres de las clases
            class_names.append(class_folder.name)
            print(f"\nCargando clase: {class_folder.name}")
            
            # Itera y hace yield de las .jpg
            for img_path in class_folder.glob('*.jpg'):
                # Cargar imagen
                img = keras.preprocessing.image.load_img(
                    img_path,
                    # Redimensionar a 64x64, porsiacaso 
                    target_size=(64, 64)
                )
                # Convertir a array numpy 
                # 64: dimension 1,64: dimension 2, 3: RGB)
                img_array = keras.preprocessing.image.img_to_array(img)
                # Agregar a listas respectivas
                images.append(img_array)
                labels.append(class_idx)
    
    # MAX_IMGS = 50

    # for class_idx, class_folder in enumerate(sorted(data_dir.iterdir())):
    #     if class_folder.is_dir():
    #         class_names.append(class_folder.name)
    #         print(f"\nCargando clase: {class_folder.name}")

    #         count = 0  # contador por clase

    #         for img_path in class_folder.glob('*.jpg'):
    #             if count >= MAX_IMGS:
    #                 break  # detener después de 50
                
    #             img = keras.preprocessing.image.load_img(
    #                 img_path,
    #                 target_size=(64, 64)
    #             )
    #             img_array = keras.preprocessing.image.img_to_array(img)

    #             images.append(img_array)
    #             labels.append(class_idx)

    #             count += 1


    # Convertir a arrays numpy
    # Cada imagen es un array Nx64x64x3, con N = num imagenes
    X = np.array(images)
    # Etiquetas como array N, de 0-4
    y = np.array(labels)
    
    print(f"\nTotal de imagenes cargadas: {len(X)}")
    print(f"Forma de las imagenes: {X.shape}")
    print(f"Clases: {class_names}")
    print(f"Distribución por clase: {np.bincount(y.astype('int'))}")
    
    # Normalizacion: escalar pixeles a rango [0, 1]
    X = X.astype('float32') / 255.0
    print("\nImágenes normalizadas al rango [0, 1]")
    
    # División estratificada del dataset
    # Separar conjunto de prueba (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.15, 
        stratify=y, 
        random_state=42
    )
    
    # Del 85% restante, separar validacion (15% del total ≈ 17.6% del temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.176,
        stratify=y_temp,
        random_state=42
    )
    
    print("\nDistribucion:\n")
    print(f"Entrenamiento: {len(X_train)} imagenes ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validación:    {len(X_val)} imagenes ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Prueba:        {len(X_test)} imagenes ({len(X_test)/len(X)*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


# 2. Construccion de la arquitectura CNN
# Dimensuiones de entrada: 64x64x3 (RGB)
# Numero de clases: 5 (0-4)
def build_cnn_model(input_shape=(64, 64, 3), num_classes=5):
    
    """
    Construye una arquitectura CNN optimizada para clasificacion de imagenes.
    
    Justificacion de la arquitectura:
    
    Bloque 1 (32 filtros, 3x3):
    - Detecta caracteristicas basicas (bordes, texturas simples)
    - 32 filtros son suficientes para imagenes de 64x64
    
    Bloque 2 (64 filtros, 3x3):
    - Captura patrones mas complejos (combinaciones de bordes)
    - Duplicamos filtros al reducir dimension espacial
    
    Bloque 3 (128 filtros, 3x3):
    - Aprende caracteristicas de alto nivel especificas de cada clase
    - Mayor capacidad representacional
    
    MaxPooling (2x2):
    - Reduce dimensionalidad espacial
    - Proporciona invariancia a pequeñas traslaciones
    - Reduce costo computacional
    
    Fully Connected Layers:
    - 256 neuronas: suficiente capacidad sin sobreajuste
    - 5 neuronas de salida (una por clase)
    """
    
    print("\n2. Construccion de la arquitectura CNN\n")
 
    # Capas de la CNN apiladas secuencialmente
    model = models.Sequential([
        # Bloque 1
        # Capa convulucional 1
        # 32 filtros de 3x3, activacion ReLU
        # Padding 'same' para mantener dimensiones
        # Input shape definida en la primera capa
        # MaxPooling 2x2 para reducir a la mitad
        layers.Conv2D(32, (3, 3), activation='relu', 
                      padding='same', input_shape=input_shape,
                      name='conv1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', 
                      padding='same', name='conv1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        # Output: 32x32x32

        # Bloque 2
        # Mismo que anterior, con 32 filtros adicionales
        layers.Conv2D(64, (3, 3), activation='relu', 
                      padding='same', name='conv2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', 
                      padding='same', name='conv2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        # Output: 16x16x64
        
        # Bloque 3
        # Mismo que anterior, con 64 filtros adicionales
        layers.Conv2D(128, (3, 3), activation='relu', 
                      padding='same', name='conv3_1'),
        layers.Conv2D(128, (3, 3), activation='relu', 
                      padding='same', name='conv3_2'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        # Output: 8x8x128
        
        # Capas Fully Connected
        # Flatten convierte 3D a 1D, en vector
        layers.Flatten(name='flatten'),
        # Dense capa con 256 neuronas conectadas y ReLU
        layers.Dense(256, activation='relu', name='fc1'),
        # Capa de salida con softmax para clasificacion multiclase
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    print("\nArquitectura construida exitosamente")
    print("\nResumen de la arquitectura:\n")
    model.summary()
    
    return model


# 3. Hiperparámetros y justificación

def get_hyperparameters():
    
    """
    Retorna y justifica los hiperparámetros seleccionados.
    
    Justificacion:
    
    1. Learning rate = 0.001
       - Valor estandar para Adam optimizer
       - Permite convergencia estable sin oscilaciones
       - Suficientemente pequeño para ajuste fino
    
    2. Batch size = 32
       - Balance entre eficiencia y generalizacion
       - Permite actualizacion frecuente de pesos
       - Ajustado al tamaño del dataset
    
    3. epochs = 50
       - Suficiente para convergencia en datasets pequeños-medianos
       - Se usara Early Stopping para evitar sobreentrenamiento
       - Monitoreo de validacion cada epoca
    
    4. Kernel size = 3x3
       - Estandar en CNNs modernas (VGG, ResNet)
       - Captura patrones locales efectivamente
       - Computacionalmente eficiente
    
    5. Numero de filtros = [32, 64, 128]
       - Progresion estandar: duplica al reducir dimension
       - Balance entre capacidad y overfitting
       - Apropiado para 5 clases
    
    6. Optimizer = Adam
       - Adaptive learning rate
       - Combina momentum y RMSprop
       - Mejor performance empírica que SGD
    """
    
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'kernel_size': (3, 3),
        'filters': [32, 64, 128],
        'dense_units': 256,
        'optimizer': 'adam'
    }
    
    print("\n3. Hiperparámetros y justificación\n")
    
    print("\nHiperparámetros seleccionados:\n")

    for key, value in hyperparams.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
    
    return hyperparams


# 4. Compilacion y entrenamiento del modelo
def compile_and_train(model, train_data, val_data, hyperparams):
    
    """
    Compila y entrena el modelo con los hiperparametros especificados.
    Utiliza el conjunto de validacion para monitorear el desempeño.
    """

    print("\n4. Compilacion y entrenamiento del modelo\n")

    # Datos de entrenamiento y validacion
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Compilar modelo
    # .compile() configura el modelo para entrenamiento
    # optimizer Adam con learning rate especificado
    # loss es la funcion de perdida para medir error
    # metrics son las metricas a monitorear
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModelo compilado")
    print(f"Optimizer: Adam (lr={hyperparams['learning_rate']})")
    print(f"Loss: Sparse Categorical Crossentropy")
    print(f"Metrics: Accuracy")
    
    # Callbacks para mejorar entrenamiento
    callbacks = [
        # Early Stopping: detiene si validacion no mejora
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # ReduceLROnPlateau: reduce learning rate si se estanca
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\nIniciando entrenamiento")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"Epochs maximas: {hyperparams['epochs']}")
    print(f"Early stopping: activado (patience=10)")

    
    # Entrenar modelo
    # fit entrena el modelo con los datos dados
    # X_train, y_train: datos de entrenamiento
    # batch_size: numero de muestras por actualizacion
    # epochs: numero maximo de epocas
    # validation_data: datos para validacion
    # callbacks: funciones para mejorar entrenamiento
    # verbose=1: mostrar progreso
    history = model.fit(
        X_train, y_train,
        batch_size=hyperparams['batch_size'],
        epochs=hyperparams['epochs'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEntrenamiento completado")
    
    return history


# 5. Evaluacion en conjunto de prueba
def evaluate_model(model, test_data, class_names):

    """
    Evalua el modelo en el conjunto de prueba y presenta metricas.
    """
    
    print("\n5. Evaluacion en conjunto de prueba\n")
    
    # Datos de prueba
    X_test, y_test = test_data
    
    # Evaluación general
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nMetricas y desempeño:\n")
    print(f"- Test Loss: {test_loss:.4f}")
    print(f"- Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Predicciones para metricas detalladas
    # predict devuelve probabilidades por clase
    # argmax obtiene la clase con mayor probabilidad
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Matriz de confusion
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nReporte de clasificacion:\n")
    # Imprime precision, recall, f1-score por clase
    print(classification_report(y_test, y_pred_classes, 
                                target_names=class_names,
                                digits=4))
    
    print("\nMatriz de confusion:\n")
    # Genera e imprime la matriz de confusion
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    return test_accuracy, y_pred_classes


# 6. Visualizacion de curvas de aprendizaje
def plot_training_curves(history):

    """
    Genera y muestra las curvas de entrenamiento y validacion.
    Incluye accuracy y loss por época.
    """
    
    print("\n6. Visualizacion de curvas de aprendizaje")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Curva de Accuracy
    axes[0].plot(history.history['accuracy'], label='Entrenamiento', 
                 linewidth=2, marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Validación', 
                 linewidth=2, marker='s', markersize=4)
    axes[0].set_title('Accuracy vs Época', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Curva de Loss
    axes[1].plot(history.history['loss'], label='Entrenamiento', 
                 linewidth=2, marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Validación', 
                 linewidth=2, marker='s', markersize=4)
    axes[1].set_title('Loss vs Época', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curvas_entrenamiento_base.png', dpi=300, bbox_inches='tight')
    print("\nGraficas guardadas: 'curvas_entrenamiento_base.png'")
    plt.show()
    
    # Interpretacion
    print("\nInterpretacion de las curvas:\n")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    
    print(f"- Accuracy final (entrenamiento): {final_train_acc:.4f}")
    print(f"- Accuracy final (validación): {final_val_acc:.4f}")
    print(f"- Gap (diferencia): {gap:.4f}")
    
    if gap > 0.1:
        print("\nSobreajuste detectado:")
        print("- Gap > 10% indica overfitting")
        print("- El modelo memoriza datos de entrenamiento")
        print("- Solucion: añadir dropout (Actividad 2)")
    elif gap < 0.05:
        print("\nBuen Ajuste:")
        print("- Gap < 5% indica buen balance")
        print("- El modelo generaliza correctamente")
    else:
        print("\nAjuste moderado:")
        print("- Gap entre 5-10% es aceptable")
        print("- Puede mejorarse con regularizacion")


# Main
def main():

    """
    Ejecuta el pipeline completo de la Actividad 1.
    """
    
    # CAMBIAR ESTA RUTA A LA UBICACIÓN DE TU DATASET
    DATA_PATH = 'dataset'  # <-- MODIFICAR AQUÍ
    
    # 1. Cargar y organizar dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = \
        load_and_organize_dataset(DATA_PATH)
    
    # 2. Construir arquitectura
    model = build_cnn_model(input_shape=(64, 64, 3), num_classes=len(class_names))
    
    # 3. Obtener hiperparametros
    hyperparams = get_hyperparameters()
    
    # 4. Entrenar modelo
    history = compile_and_train(
        model, 
        (X_train, y_train), 
        (X_val, y_val), 
        hyperparams
    )
    
    # 5. Evaluar en conjunto de prueba
    test_accuracy, predictions = evaluate_model(
        model, 
        (X_test, y_test), 
        class_names
    )
    
    # 6. Visualizar curvas de aprendizaje
    plot_training_curves(history)
    
    # Guardar modelo
    model.save('modelo_cnn_base.h5')
    print("\nModelo guardado: 'modelo_cnn_base.h5'")
    
    return model, history

if __name__ == "__main__":
    model, history = main()