# Transformadas de scattering: invarianza sin aprendizaje

Trabajo final del curso de Matemáticas del aprendizaje de máquinas. Exposición de las *scattering
transforms* de Bruna y Mallat, réplica parcial de sus experimentos sobre MNIST y
comparación empírica contra una CNN entrenada end-to-end.

## Pregunta central

La transformada de scattering es una cascada de convoluciones wavelet, módulo y
promediado con filtros fijos y cero parámetros aprendidos. La hipótesis que
pone a prueba este trabajo es que esa ausencia de aprendizaje debería ser una
ventaja con pocos datos, donde una CNN tiene demasiada
capacidad para la evidencia disponible. El experimento central es por tanto la
curva de precisión frente a tamaño del conjunto de entrenamiento. 

## Video de sustentacion

https://drive.google.com/file/d/1cVHKfNbJDfTuMKI2Htu7jnaFPvu8_PVF/view?usp=sharing

## Estructura

```
src/          lógica reutilizable (datos, scattering, modelos, evaluación)
notebooks/    exploración y figuras; llaman a src/, no duplican lógica
results/      métricas en JSON/CSV, versionadas (los cachés pesados no)
figures/      salida vectorial PDF/SVG para el documento
paper/        fuentes LaTeX
```

## Entorno

PyTorch y Kymatio no publican ruedas para Python 3.14, así que el proyecto usa un
entorno aislado con Python 3.12 gestionado por [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate          # Windows
uv pip install -r requirements.txt
```

## Reproducibilidad

Cada experimento fija semilla vía `src.repro.set_seed` y adjunta a sus
resultados una instantánea del entorno (versiones de librerías, GPU, commit de
git) mediante `src.repro.environment_report`. Los resultados intermedios se
guardan en `results/` para poder iterar la redacción sin re-entrenar.


