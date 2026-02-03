# LLMs Comparison - Proyecto de Navegación Autónoma en CARLA

Este proyecto implementa y evalúa diferentes modelos de lenguaje multimodal (LMMs) para la tarea de navegación autónoma en intersecciones utilizando el simulador CARLA. El sistema combina imágenes BEV (Bird's Eye View) y frontales para predecir la dirección correcta (Straight, Right, Left) que debe tomar un vehículo autónomo.

---

## 📁 Estructura del Proyecto

### `dataset/`

Contiene el dataset limpio tras la revisión individualizada de cada muestra. La estructura es la siguiente:

- **`data.json`**: Archivo JSON principal que contiene todas las muestras del dataset. Para cada muestra se incluye:
  - Rutas a las 4 imágenes (frontal RGB, frontal segmentada, BEV RGB, BEV segmentada)
  - Coordenadas de origen y destino en píxeles
  - Ground truth (dirección correcta)
  - Weather empleado (solo afecta a las imágenes sin segmentar)

- **`Simulation_scripts/dataset/front/`**: Almacena las imágenes frontales del vehículo (RGB y segmentadas)

- **`Simulation_scripts/dataset/bev/`**: Almacena las imágenes BEV (Bird's Eye View) del mapa

- **Imágenes especiales (`clean_bev_ss_image`)**: Mapas BEV segmentados vacíos (sin indicadores de origen/destino). Estas imágenes sirvieron como base para posteriormente agregar los indicadores visuales de origen (flecha roja) y destino (cuadrado azul).

---

### `Simulation_scripts/`

Contiene los scripts para la generación del dataset utilizando el simulador CARLA.

#### `Simulation_scripts/gross_dataset/`

Directorio con todas las muestras generadas, aún sin revisar manualmente. Incluye:

- **`intersections.json`**: Contiene la misma información que `data.json`, pero con información de depuración extendida para facilitar el análisis y filtrado de muestras.

#### Scripts principales

- **`carla_agent_screenless.py`**: Script principal que controla el agente autónomo en CARLA. Se encarga de:
  - Inicializar el vehículo ego y los sensores (cámaras RGB y de segmentación semántica)
  - Capturar imágenes frontales y BEV durante la navegación
  - Detectar intersecciones y guardar automáticamente las muestras
  - Generar las anotaciones con las coordenadas de origen/destino y el ground truth

- **`mc_min_screenless.py`**: Módulo auxiliar que proporciona funciones para inicializar el entorno de CARLA y gestionar el bucle de juego sin interfaz gráfica (modo headless).

- **`mc_utils_screenless.py`**: Utilidades adicionales para la gestión del mundo de CARLA, incluyendo funciones para obtener el vehículo ego y cerrar correctamente la simulación.

- **`routes.json`**: Archivo de configuración que define las rutas de navegación. Las rutas se generan aleatoriamente para aumentar la variabilidad del dataset y cubrir diferentes escenarios de intersecciones.

---

### `Test_scripts/`

Contiene los scripts y recursos necesarios para evaluar los modelos LMM en el dataset de CARLA.

#### `Test_scripts/prompts/`

Directorio que almacena los 4 prompts necesarios para ejecutar las pruebas. Cada prompt se corresponde con una configuración de entrada específica:

- **Prompt BEV**: Solo imagen BEV segmentada
- **Prompt BEV + Frontal**: Imagen BEV segmentada + imagen frontal segmentada
- **Prompt BEV + Coords**: Imagen BEV segmentada + coordenadas de origen y destino
- **Prompt BEV + Frontal + Coords**: Imagen BEV segmentada + imagen frontal segmentada + coordenadas

#### Scripts principales

- **`test_carla.py`**: Script principal de evaluación que:
  - Carga el dataset y lo divide en conjuntos de entrenamiento y validación
  - Ejecuta las diferentes configuraciones de prueba sobre los modelos
  - Calcula métricas de rendimiento (accuracy, F1-score, precision, recall, matriz de confusión)
  - Guarda los resultados y las respuestas del modelo en formato JSON
  - Identifica y guarda las muestras mal clasificadas para análisis posterior

- **`models_api.py`**: Módulo que proporciona una API unificada para cargar y ejecutar diferentes modelos LMM:
  - Funciones `load_lmm()` para cargar modelos base y fine-tuned (LoRA/DoRA)
  - Funciones `call_lmm()` para realizar inferencia con soporte para In-Context Learning (ICL)
  - Soporte para múltiples arquitecturas: LLaVA, Gemma, Qwen, InternVL, Gemini API
  - Gestión automática de distribución multi-GPU y cuantización

---

### `CIL_adaptations/`

Contiene las adaptaciones del modelo CIL++ ([Conditional Imitation Learning++](https://arxiv.org/pdf/2302.03198)) para este proyecto.

#### Arquitectura de CIL++

CIL++ es una arquitectura de aprendizaje por imitación condicional diseñada para conducción autónoma. El modelo:

- Utiliza una red neuronal convolucional (CNN) como encoder visual para extraer características de las imágenes
- Implementa un mecanismo de atención para fusionar información de múltiples vistas
- Predice acciones de control (steering, throttle, brake) condicionadas a comandos de alto nivel
- En este proyecto, se ha adaptado para predecir directamente la dirección en intersecciones (clasificación)

#### Archivos

- **`CIL_singleview.py`**: Adaptación de CIL++ para la configuración donde el modelo **solo recibe imágenes BEV segmentadas**. Esta versión simplificada procesa únicamente la vista aérea para tomar decisiones.

- **`CIL_binaryview.py`**: Adaptación de CIL++ para la configuración donde el modelo recibe **tanto la imagen BEV como la imagen frontal**. 
  - Se puede configurar si se desean imágenes segmentadas o RGB modificando la variable `SEGMENTED_FRONT` al cargar el dataset
  - Implementa fusión de características de ambas vistas mediante concatenación o atención

**Nota**: Estos archivos requieren de toda la estructura del proyecto CIL++ para ejecutarse correctamente. Es necesario tener instaladas las dependencias específicas de CIL++ y la estructura de directorios completa del repositorio original.

---

### `Improvements/`

Contiene mejoras y experimentos adicionales sobre los modelos base.

#### `Improvements/PEFT/`

Directorio dedicado al fine-tuning de modelos mediante **DoRA (Weight-Decomposed Low-Rank Adaptation)** utilizando la librería [PEFT de HuggingFace](https://github.com/huggingface/peft).

DoRA es una variante mejorada de LoRA que descompone los pesos en magnitud y dirección, logrando mejor rendimiento con el mismo número de parámetros entrenables.

#### Flujo de trabajo para fine-tuning

1. **`create_dataset.py`**: Envuelve el dataset de CARLA en el formato de la API `Datasets` de HuggingFace, creando un objeto `Dataset` compatible con los pipelines de entrenamiento.

2. **`create_peft_wrapper.py`**: Prepara el modelo y los datos para el entrenamiento PEFT:
   - Tokeniza correctamente los prompts según el formato de chat de cada modelo
   - Procesa las imágenes con el processor correspondiente
   - Configura los parámetros de DoRA (rank, alpha, target modules)
   - Crea el wrapper PEFT sobre el modelo base

3. **`train_model.py`**: Ejecuta el entrenamiento de tipo SFT (Supervised Fine-Tuning):
   - Utiliza `SFTTrainer` de la librería `trl` de HuggingFace
   - Implementa callbacks personalizados para logging y checkpointing
   - Guarda el modelo fine-tuned y los adaptadores LoRA/DoRA

#### Archivos adicionales

- **`get_linear_layers.py`**: Utilidad para identificar automáticamente las capas lineales del modelo que serán objetivo del fine-tuning con PEFT.

- **`trainer_callback.py`**: Define callbacks personalizados para el proceso de entrenamiento (e.g., logging de métricas, guardado de checkpoints intermedios).

---

### `requirements.txt`

Archivo con todas las dependencias del proyecto. Para instalar las librerías necesarias, ejecuta:

```bash
pip install -r requirements.txt
```

**Nota**: Algunas dependencias específicas de CARLA pueden requerir instalación manual. Consulta la [documentación oficial de CARLA](https://carla.readthedocs.io/) para más detalles.

---

## 🚀 Uso Rápido

### 1. Generar dataset en CARLA

```bash
cd Simulation_scripts
python carla_agent_screenless.py --map Town01_Opt --weather ClearNoon
```

### 2. Evaluar un modelo

```bash
cd Simulation_scripts
python ../Test_scripts/test_carla.py --model "google/gemma-3-12b-it" --do-tests test_bev test_bev_frontal
```

### 3. Fine-tuning con PEFT

```bash
cd Improvements/PEFT
python create_dataset.py
python create_peft_wrapper.py
python train_model.py
```

---

## 📊 Resultados

Los resultados de las evaluaciones se guardan en `Test_scripts/test_results/{model_name}/`:

- `results.json`: Métricas de rendimiento (accuracy, F1, precision, recall, confusion matrix)
- `answers.json`: Respuestas completas del modelo para cada muestra
- `wrong_classified/`: Imágenes de los casos mal clasificados para análisis

---

## 📝 Citas

Si utilizas este código, por favor cita el paper de CIL++:

```bibtex
@article{cilplusplus2023,
  title={Conditional Imitation Learning++},
  author={...},
  journal={arXiv preprint arXiv:2302.03198},
  year={2023}
}
```

---

## 📧 Contacto

Para preguntas o colaboraciones, contacta con el equipo de desarrollo del proyecto.
