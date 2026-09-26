# Predicción y Agrupamiento de Tiempos de Entrega en E-commerce (Olist)

Este repositorio contiene el desarrollo de un proyecto de minería de datos aplicado a los registros de Olist Store, una plataforma brasileña de comercio electrónico. El objetivo principal es mejorar la logística y la experiencia del cliente mediante la predicción de tiempos de entrega y la segmentación de pedidos logísticos.   

🎯 Objetivos del Proyecto

Clustering (Descriptivo): Agrupar las transacciones según su tamaño, valor y complejidad logística para priorizar el seguimiento operativo.  
Regresión (Predictivo): Estimar con la mayor exactitud posible los días de entrega de un pedido en el momento de la compra.   

🛠️ Metodología (CRISP-DM)

El proyecto se desarrolló siguiendo las fases de la metodología CRISP-DM:   
1. Entendimiento del Negocio: Definición de objetivos para optimizar la logística centralizada de Olist frente a múltiples vendedores y marketplaces.   
2. Entendimiento de los Datos: Exploración de un dataset histórico de pedidos (2016-2018), analizando variables comerciales, físicas (peso/volumen), temporales y geográficas.   
3. Preparación de los Datos: Creación de una base agregada a nivel de pedido, ingeniería de características (cálculo de distancias e historiales logísticos recientes), limpieza y selección de variables.   
4. Modelamiento:
  Agrupamiento: Entrenamiento de un modelo K-Means utilizando 5 variables clave (distancia, estado, flete, volumen y peso).
  Regresión: Evaluación de algoritmos como Decision Tree, Random Forest, MLP y XGBoost. Se seleccionó XGBoost por su superioridad predictiva (MAE: 2.88) y eficiencia computacional.

Evaluación: Implementación de un margen de seguridad lógico (+3 días, equivalente al MAE) sobre la predicción base para asegurar que ~82% de las entregas cumplan con el tiempo prometido al cliente.   
Despliegue: Exportación de los modelos (pickle) y creación de una aplicación web interactiva en Streamlit que simula el estado histórico de los envíos y clasifica/predice la entrega de nuevas compras.   

📊 Resultados

Modelo de Clustering: Identificó tres perfiles logísticos accionables:   
1. Local y Ligero (45.4%): Prioridad baja. Distancias cortas (mayoría SP) y paquetes pequeños.
2. Voluminoso y Pesado (36.0%): Prioridad media. Paquetes grandes con mayor riesgo de daño durante la manipulación.   
3. Larga Distancia (18.5%): Prioridad alta. Trayectos largos, principalmente fuera del estado central, con alto riesgo de demora.

Modelo Predictivo
Configuración XGBoost: n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.7, colsample_bytree=0.9.   
El ajuste operativo garantiza una promesa de entrega confiable, mitigando la incertidumbre del cliente final.   

👥 Autores:

Mauricio Andrés Blanco Montero   
Luz Andrea Amaya Guzmán   
