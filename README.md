# Aplicación de Neural Simulated Annealing
Repositorio para ICS2985: Investigación o trabajo en pregrado.

La experimentación computacional abordada en este repositorio está inspirada en el método investigado por Alvaro H.C. Correia, Daniel E. Worrall, Roberto Bondesan. Paper de los autores disponible en el siguiente enlace: [Enlace a artículo](https://doi.org/10.48550/arXiv.2203.02201). El código presente en el repositorio fue elaborado a partir del repositorio público puesto a disposición por parte de los autores del paper.

La experimentación computacional de este repositorio busca verificar el funcionamiento y robustez de Neural Simulated Annealing al variar hiperparámetros, como el paso de avance y la temperatura inicial de funcionamiento, al entrenar la red neuronal asociada al método utilizando instancias generadas aleatoriamente (idéntico al paper original) o con instancias similares usando una base de datos de ubicaciones de pacientes ficticia. Además, se realiza la comparación del rendimiento del método al ser entrenado con ambos tipos de datos, utilizando nuevamente la base de datos de pacientes.


---
## Instalación
Asegúrate de tener Python ≥3.10 (probado con Python 3.10.11) y la última versión de `pip` (probado con 22.3.1):
```bash
pip install --upgrade --no-deps pip
```
Luego, instala PyTorch 1.13.0 con la versión apropiada de CUDA (probado con CUDA 11.7):
```bash
python -m pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
Finalmente, instala las dependencias restantes usando pip (verificar que el archivo se encuentre en el mismo directorio):
```bash
pip install -r requirements.txt
```

Para ejecutar el código, es necesario agregar el directorio raíz del proyecto a su PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD"
```
---
## Ejecución de Experimentos
En este trabajo siempre se va trabajará con el tipo de experimento `tsp_ppo` a diferencia del trabajo original que incluye varias opciones más.  
**Nota:** Para cambiar el valor de la **temperatura inicial**, hay que modificar la variable `init_temp` en los archivos `outputs/.hydra/config.yaml` y `scripts/config/experiment/tsp_ppo.yaml`.

### Entrenamiento
El archivo de ejecución principal para reproducir todos los experimentos es `main.py`. Usamos [Hydra](https://hydra.cc/) para configurar los experimentos, de modo que pueda reentrenar nuestros modelos de Neural SA de la siguiente manera.
#### Caso Entrenamiento con instancias similares
```bash
python scripts/main.py +experiment=tsp_ppo
```
**Importante:** para especificar si se quiere entrenar con instancias aleatorias o instancias similares:
1. Ir al archivo `scripts/main.py` 
2. Descomentar la línea correspondiente (línea 38 o 39) según tu elección. 
3. Configurar la dimensión del problema:
    - **Instancias aleatorias:** modificar línea 142 de `main.py`. 
    - **Instancias similares:** modificar línea 193 de `main.py`. 

#### Caso Evaluación TSPLIB
```bash
python scripts/main.py +experiment=tsp_ppo
```
Se ejecuta de la misma forma que en el primero caso. Para modificar la dimensión del problema, especificar:
```bash
python scripts/main.py +experiment=tsp_ppo ++problem_dim=<problem_dim>
```
**Nota:** Por defecto, el modelo se entrena con una dimensión de 20 nodos.


En ambos casos, el modelo entrenado se guarda en `outputs/models/tsp<problem_dim>-ppo`.
___
### Evaluación
#### Caso Entrenamiento con instancias similares
Antes de ejecutar la evaluación, verificar:
1. **Agente Correcto:** revisar el archivo `scripts/eval.py`, específicamente en la línea 133, para asegurarse que se esté usando el agente correcto.
2. **Archivo a evaluar:** El archivo que se evaluará se especifica en la línea 144 de `eval.py`. Este tiene que encontrarse en la carpeta `datasets/` para que sea válido.   

Una vez verificado todo lo anterior, ejecutar:

```bash
python scripts/eval.py +experiment=tsp_ppo
```
#### Caso Evaluación TSPLIB
Igualmente hay que verificar que el archivo que se está evaluando existe. Este se especifica en la línea 101 del archivo `scripts/eval.py`. Este archivo debe existir en la carpeta `datasets/`.  
Para evaluar:
```bash
python scripts/eval.py +experiment=tsp_ppo
```
**Nota:** En ambos casos, el archivo `eval.py` también ejecuta Vainilla Simulated Annealing (SA clásico). Los resultados se guardan en la carpeta `outputs/results/tsp` y pueden ser visualizados con el archivo `print_results.py`.

---
### Imprimir Resultados
#### Caso Entrenamiento con instancias similares
Hay que especificar la dimensión del problema que se quiere imprimir:

```bash
python scripts/print_results.py +experiment=tsp_ppo ++problem_dim=<problem_dim>
```
Para especificar la temperatura, esta debe cambiarse al final de las líneas 32 y 35 del archivo `print_results.py` (por defecto toma $1.0$)

#### Caso Evaluación TSPLIB
En este caso, el archivo `print_results.py` se encuentra en la carpeta principal:
```bash
python print_results.py +experiment=tsp_ppo
```
Hay que especificar el archivo en la línea 30 y 31 de `print_results.py`

---
## Estructura del Proyecto
```
ipre-aplicacion-neural-sa/
├── entrenamiento-instancias-similares/  # Experimentos con datos de pacientes
├── evaluacion-TSPLIB/                   # Evaluación con benchmark TSPLIB
├── requirements.txt                      # Dependencias del proyecto
└── README.md                            # Este archivo
```
---
## Autores
Trabajo elaborado por **Isidora Carmona, Francisca Correa, Javiera Santibáñez y Sebastian Sepúlveda**, estudiantes de Pregrado de Departamento de Ingeniería Industrial y Sistemas, Escuela de Ingeniería, Pontificia Universidad Católica de Chile.

Trabajo realizado bajo supervisión de **Jorge Vera**, profesor titular del Departamento de Ingeniería Industrial y Sistemas, Escuela de Ingeniería, Pontificia Universidad Católica de Chile.

---
**Pontificia Universidad Católica de Chile**

Escuela de Ingeniería

Departamento de Ingeniería Industrial y Sistemas

Diciembre, 2025.
---
## Referencias
Correia, A. H., Worrall, D. E., & Bondesan, R. (2022). Neural Simulated Annealing. arXiv preprint arXiv:2203.02201. https://doi.org/10.48550/arXiv.2203.02201
