# Scattering Transforms: teoría y réplica experimental

Trabajo final del curso de Deep Learning. Exposición de las *scattering
transforms* de Bruna y Mallat, réplica parcial de sus experimentos sobre MNIST y
comparación empírica contra una CNN entrenada end-to-end.

## Pregunta central

La transformada de scattering es una cascada de convoluciones wavelet, módulo y
promediado con **filtros fijos y cero parámetros aprendidos**. La hipótesis que
pone a prueba este trabajo es que esa ausencia de aprendizaje debería ser una
ventaja precisamente en el régimen de pocos datos, donde una CNN tiene demasiada
capacidad para la evidencia disponible. El experimento central es por tanto la
curva de **precisión frente a tamaño del conjunto de entrenamiento**, no la
precisión con el dataset completo.

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

## Referencias

- Bruna, J. & Mallat, S. (2013). *Invariant Scattering Convolution Networks*.
  IEEE TPAMI 35(8), 1872–1886. [arXiv:1203.1513](https://arxiv.org/abs/1203.1513)
- Mallat, S. (2012). *Group Invariant Scattering*. CPAM 65(10), 1331–1398.
- Bruna, J. & Mallat, S. (2011). *Classification with Scattering Operators*. CVPR.
- Andreux et al. (2020). *Kymatio: Scattering Transforms in Python*. JMLR 21(60).
