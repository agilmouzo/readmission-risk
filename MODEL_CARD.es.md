# Ficha del modelo: riesgo de reingreso a 30 días en pacientes diabéticos

[English](MODEL_CARD.md) | **Español**

Esta ficha resume qué es el modelo, con qué datos se entrenó, qué tal funciona y dónde no debe
usarse. Sigue la idea de una «ficha de modelo» (*model card*): una descripción breve y honesta que
acompaña al modelo. Los detalles y las figuras están en el [README](README.es.md) y en `reports/`.

> Proyecto educativo. **No es una herramienta clínica.** No debe usarse para tomar decisiones sobre
> pacientes reales.

## Detalles del modelo
- **Tarea:** clasificación binaria. Para una estancia hospitalaria de un paciente con diabetes,
  estimar la probabilidad de que el paciente reingrese en el hospital en los 30 días siguientes al
  alta.
- **Modelo:** árboles de decisión con gradient boosting (LightGBM, 300 árboles, tasa de aprendizaje
  0,03, 15 hojas, regularización fuerte) dentro de una pipeline de scikit-learn que también hace el
  preprocesado. No se usan pesos de clase, de modo que las probabilidades no se inflan
  artificialmente.
- **Versión:** `v0.1.0`, publicada como archivo de una Release de GitHub (`model.joblib`, menos de
  1 MB). El archivo se comprueba con una huella SHA-256 al descargarlo, y `GET /health` indica qué
  versión está cargada. Semilla aleatoria 42.
- **Entradas:** la información disponible cuando el paciente sale del hospital: edad, sexo, cómo
  ingresó y cómo se le dio el alta, duración de la estancia, número de pruebas, procedimientos y
  medicamentos, visitas del año anterior, diagnósticos (agrupados a partir de códigos ICD-9 en
  grupos generales), resultados de HbA1c y glucosa, medicación para la diabetes, tipo de seguro y
  especialidad médica.
- **Salida:** una probabilidad entre 0 y 1, y un indicador `high_risk` que es verdadero cuando la
  probabilidad supera un umbral fijo (0,1971).
- **Responsable:** el autor del repositorio (GitHub: agilmouzo). Proyecto personal de portfolio,
  licencia MIT.

## Uso previsto
- **Dentro del alcance:** aprendizaje, demostración y debate sobre cómo construir, evaluar, explicar
  y servir un modelo de riesgo con cuidado.
- **Fuera del alcance:** cualquier decisión clínica, operativa o administrativa sobre pacientes
  reales; su uso con datos de otros países o sistemas de salud (incluido el español); su uso con
  pacientes dados de alta por fallecimiento o a cuidados paliativos (la API rechaza esos códigos); su
  uso antes del alta.

## Datos de entrenamiento
- **Origen:** *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository
  (Clore, Cios, DeShazo y Strack, 2014; CC BY 4.0). 101.766 estancias hospitalarias de 71.518
  pacientes en 130 hospitales de EE. UU.
- **Limpieza:** se eliminan las estancias que terminaron en fallecimiento o en cuidados paliativos
  (esos pacientes no pueden reingresar). Los datos no se guardan en este repositorio.
- **Objetivo:** reingreso en menos de 30 días (alrededor del 11 % de las estancias). Los datos no
  permiten saber si un reingreso fue planificado, evitable o relacionado con la atención recibida.
- **División:** por paciente, de modo que las estancias de una misma persona nunca aparecen a la vez
  en entrenamiento y en test. 80 % entrenamiento, 20 % test (19.802 estancias).
- **No se usan como entradas:** la raza (se guarda solo para comparar el rendimiento entre grupos),
  los identificadores de paciente y de ingreso, y el peso, que casi siempre falta.

## Rendimiento
Conjunto de test, con intervalos de confianza al 95 % obtenidos remuestreando pacientes:

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,676 (0,660–0,690) |
| AUC-PR (aleatorio = 0,112) | 0,234 (0,211–0,257) |
| Puntuación de Brier | 0,094 (0,090–0,098) |
| Tasa de reingreso en el 10 % de pacientes de mayor riesgo | 28,5 % frente a 11,2 % del total (lift 2,5×) |

Con el umbral de alto riesgo, el modelo marca al 9,7 % de los pacientes de test, con precisión del
28,6 % y recall del 24,9 %. La calibración es buena para el grupo en conjunto (pendiente 1,06,
error de calibración esperado 0,007), pero eso no hace que la cifra de un paciente concreto sea
igual de precisa. Frente a una regresión logística la mejora es pequeña (unos 0,01 de AUC-ROC).

## Rendimiento por subgrupos
La discriminación es parecida por género y entre los dos grupos grandes de raza (AUC-ROC 0,667 en
afroamericanos y 0,675 en caucásicos, con intervalos que se solapan). Es claramente menor en
pacientes de 80 años o más (0,632) que en menores de 40 (0,777). Los grupos pequeños tienen muy
pocos casos para sacar conclusiones. Una diferencia de AUC por sí sola no demuestra trato injusto,
y una diferencia pequeña no demuestra su ausencia. Tablas completas: `reports/subgroups.csv`.

## Limitaciones y riesgos
- Datos antiguos y solo de EE. UU. (1999–2008). La práctica, la codificación y el perfil de
  pacientes han cambiado. No hay validación temporal ni prueba en hospitales externos.
- Poder predictivo modesto: la mayoría de los reingresos no se marcan y la mayoría de las marcas
  son falsas alarmas. Ordena mejor a grupos de pacientes de lo que predice a individuos.
- Predice solo en el momento del alta.
- Dejar fuera la raza no elimina los indicadores indirectos: el tipo de seguro y la especialidad
  médica pueden llevar información socioeconómica.
- Las explicaciones (SHAP) muestran asociaciones aprendidas de los datos, no causas.
- Sin un ajuste exhaustivo. La API no tiene autenticación ni límite de peticiones.

## Consideraciones éticas
El modelo predice un resultado relacionado con la salud y, si se usara mal, podría perjudicar a
grupos de pacientes, por ejemplo por edad o por tipo de seguro. Por eso se publica solo como un
ejemplo educativo, con sus límites indicados junto a los resultados y en la demo.

## Cómo citar y contacto
Código y modelo: licencia MIT, https://github.com/agilmouzo/readmission-risk. Si usas los datos,
cita el conjunto de datos (https://doi.org/10.24432/C5230J) y el estudio original (Strack et al.,
2014, *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical
Database Patient Records*, BioMed Research International).
