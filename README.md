# Proyecto de Grado — Reconocimiento Facial

Sistema de reconocimiento facial en tiempo real usando OpenCV LBPH y una webcam.

## Estructura del proyecto

```
Proyecto de grado/
├── src/
│   ├── main.py            # Punto de entrada: bucle de cámara y detección
│   ├── recognizer.py      # Lógica de reconocimiento con el modelo LBPH
│   └── data_collector.py  # Extrae imágenes de entrenamiento desde un video
├── models/
│   ├── modelo_face.xml    # Modelo LBPH entrenado
│   ├── label_encoder.pkl  # Codificador de etiquetas (nombres de personas)
│   └── cascades/
│       └── haarcascade_frontalface_alt2.xml
├── data/                  # Imágenes de entrenamiento por persona
│   ├── Aharu Chacon/
│   ├── Alejandro Gama/
│   ├── Claudia Aguilar/
│   └── Samuel Jimenez/
├── logs/
├── requirements.txt
└── .gitignore
```

## Instalación

**1. Crear y activar el entorno virtual**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**2. Instalar dlib (Windows — requiere el .whl precompilado)**
```bash
pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
```

**3. Instalar el resto de dependencias**
```bash
pip install -r requirements.txt
```

## Uso

**Ejecutar el reconocimiento facial en tiempo real:**
```bash
python src/main.py
```
Presiona `q` para salir.

**Recolectar imágenes de entrenamiento desde un video:**
Edita `VIDEO_PATH` y `PERSON_NAME` al final de `src/data_collector.py`, luego:
```bash
python src/data_collector.py
```

## Umbral de confianza

El umbral de confianza LBPH se configura en `src/recognizer.py`:
```python
CONFIDENCE_THRESHOLD = 60  # Menor valor = más estricto
```

## Notas

- El modelo fue entrenado con imágenes de 200×200 píxeles en escala de grises.
- Si se agregan nuevas personas, el modelo debe reentrenarse.
- `dlib` no se instala con `pip install -r requirements.txt` en Windows; usar el `.whl` directamente.
