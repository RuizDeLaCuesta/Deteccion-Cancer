''' DESARROLLO DE UN MODELO PREDICTIVO DE CÁNCER MEDIANTE LA ANAMNESIS Y OTROS DATOS CLÍNICOS '''

## Importación de Librerías Necesarias para el Desarrollo#

import pandas as pd 
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

## Carga del Archivo 'csv' mediante Path Relativo ##

path = r'C:\Users\ruizd\CEI\Archivos\The_Cancer_data_1500_V2.csv'
cancer = pd.read_csv(path)

## Análisis Explotatorio de Datos (EDA) ##

cancer.head()
cancer.tail()
cancer.sample(20)
cancer.info()
cancer.isnull().sum()

# Creación de Nueva Columna

cancer['ID'] = range(1, len(cancer) + 1)

# Conversión de la Columna a Índice del DataFrame

cancer.set_index('ID', inplace=True)

# Exportado Nuevo DataFrame

cancer.to_csv('cancer_detecion.csv', index=True, encoding='utf-8')

## Normalización de Datos por Columnas ##
# MinMax (valores entre 0 y 1)

from sklearn.preprocessing import MinMaxScaler
minmax_normalizador = MinMaxScaler()
cancer_minmax_norm = pd.DataFrame(minmax_normalizador.fit_transform(cancer))
print(cancer_minmax_norm.head())
cancer_minmax_norm.to_csv('cancer_detection_MinMaxNormalized.csv', index=True, encoding='utf-8')

# Standard (media de 0 y 1 de desviacion)

from sklearn.preprocessing import StandardScaler
std_normalizador = StandardScaler()
cancer_std_norm = pd.DataFrame(std_normalizador.fit_transform(cancer))
print(cancer_std_norm.head())
cancer_std_norm.to_csv('cancer_detection_StandardNormalized.csv', index=True, encoding='utf-8')

##  Análisis Estadístico Descriptivo de Datos ##

# Función para Contabilizar Aspectos Afirmativos / Negativos

def contadores(columna):
    contador = 0
    for i in columna:
        if i == 1 or i == 2:
            contador += 1
    print(contador)

contadores(cancer['Diagnosis'])
contadores(cancer['Smoking'])
contadores(cancer['CancerHistory'])
contadores(cancer['GeneticRisk'])

# Graficación de Pacientes (fumadores, r.genético, p.canceroso, diagnosticados)

caracteristicas = [
    'Total Pacientes', 
    'Fumadores', 
    'Historial Canceroso', 
    'Riesgo Genético', 
    'Diagnosticados'
]
valores = [1500, 404, 216, 605, 557]

plt.figure(figsize=(9, 5))
plt.bar(caracteristicas, valores, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title('Distribución Pacientes Categorías Binarias')
plt.xlabel('ATRIBUTOS')
plt.ylabel('NÚMERO PACIENTES')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Función de Descripción Estadística

def analisis_est(columna, nombre_columna):
    print(f'La media de {nombre_columna} es {columna.mean()}')
    print(f'La mediana de {nombre_columna} es {columna.median()}')
    print(f'La desviación estándar de {nombre_columna} es {columna.std()}')
    print(f'La asimetría en los valores de {nombre_columna} es de {columna.skew()}')
    print(f'El cuartil inferior de {nombre_columna} es de {columna.quantile(0.25)}')
    print(f'El cuartil medio de {nombre_columna} es de {columna.quantile(0.5)}')
    print(f'El cuartil superior de {nombre_columna} es de {columna.quantile(0.75)}')

analisis_est(cancer['Age'], 'Edad')
analisis_est(cancer['BMI'], 'IMC')
cancer['Age'].describe()

## Investigación de Algoritmos para el Caso de Uso ##

# Etiquetas = cancer['Diagnosis'] / 0 = no cancer 1 = cancer

# División de Datos en Valores Axiales
x = cancer.drop('Diagnosis', axis=1)
y = cancer['Diagnosis']

# División en conjuntos de entrenamiento y prueba

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Breve revisión de los nuevos conjuntos de datos mediante slicing

print(x_train[:5], y_train[:5])
print(x_test[:5], y_test[:5])

# Primeros modelos 

# 1. KNN

from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(x_train, y_train)
y_pred_knn= knn1.predict(x_test)
knn1

# 2. Árbol de Decisiones

from sklearn.tree import DecisionTreeClassifier
dt1 = DecisionTreeClassifier()
dt1.fit(x_train, y_train)
y_pred_dt = dt1.predict(x_test)

# 3. Bosque Aleatorio

from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(n_estimators=100)
rf1.fit(x_train, y_train)
y_pred_rf = rf1.predict(x_test)

# 4. Máquina Soporte Vectorial

from sklearn.svm import SVC
svm1 = SVC()
svm1.fit(x_train, y_train)
y_pred_svm = svm1.predict(x_test)

# 5. Regresión Logística

from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(max_iter=100000)
lr1.fit(x_train, y_train)
y_pred_lr = lr1.predict(x_test)

# Función de reporte de rendimiento de los modelos de clasificación

from sklearn.metrics import classification_report
def reporte_clasificacion(prediccion, nombre_modelo):
    reporte = classification_report(y_test, prediccion)

    print(f'Reporte de Clasificación del Modelo {nombre_modelo} \n {reporte}')

reporte_clasificacion(y_pred_knn, 'KNN')
reporte_clasificacion(y_pred_dt, 'DT')
reporte_clasificacion(y_pred_rf, 'RF')
reporte_clasificacion(y_pred_svm, 'SVM')
reporte_clasificacion(y_pred_lr, 'LR')

# Mejora de los Modelos mediante el Ajuste de Hiperparámetros

# 1. KNN

from sklearn.neighbors import KNeighborsClassifier
knn2 = KNeighborsClassifier(n_neighbors=16)
knn2.fit(x_train, y_train)
y_pred_knn2 = knn2.predict(x_test)

# 2. Arbol de Decisiones

from sklearn.tree import DecisionTreeClassifier
dt2 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
dt2.fit(x_train, y_train)
y_pred_dt2 = dt2.predict(x_test)

# 3. Bosque Aleatorio

from sklearn.ensemble import RandomForestClassifier
rf2 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, max_features=1)
rf2.fit(x_train, y_train)
y_pred_rf2 = rf2.predict(x_test)

# 4. Máquina Soporte Vectorial

from sklearn.svm import SVC
svm2 = SVC(kernel='linear')
svm2.fit(x_train, y_train)
y_pred_svm2 = svm2.predict(x_test)

# 5. Regresión Logística

from sklearn.linear_model import LogisticRegression
lr2 = LogisticRegression(max_iter=100000, penalty='l2')
lr2.fit(x_train, y_train)
y_pred_lr2 = lr2.predict(x_test)

## Evaluación de Modelos ##

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Función medidora de accuracy, recall y precision

def rendimiento(prediccion, nombre_modelo):
    accuracy = accuracy_score(y_test, prediccion)
    recall = recall_score(y_test, prediccion)
    precision = precision_score(y_test, prediccion)

    print(f'{accuracy} de accuracy para el modelo {nombre_modelo}')
    print(f'{recall} de recall para el modelo {nombre_modelo}')
    print(f'{precision} de precision para el modelo {nombre_modelo}')

rendimiento(y_pred_knn2, 'KNN')
rendimiento(y_pred_dt2, 'DT')
rendimiento(y_pred_rf2, 'RF')
rendimiento(y_pred_svm2, 'SVM')
rendimiento(y_pred_lr2, 'LR')

# Bucles determinantes del modelo con mayor accuracy, recall y F1

accuracy_contador = {'KNN': 0.6933333333333334,
                     'DT' : 0.8366666666666667,
            'RF' : 0.8866666666666667,
            'SVM' : 0.8577777777777777,
            'LR' : 0.8633333333333333
            }
recall_contador = {'KNN': 0.35344827586206895,
                   'DT' : 0.75648728763461892,
            'RF' : 0.7327586206896551,
            'SVM' : 0.7586206896551724,
            'LR' : 0.7498484748389891
            }
precision_contador = {'KNN': 0.7068965517241379,
                      'DT' : 0.8130841121495327,
            'RF' : 0.9659090909090909,
            'SVM' : 0.8627450980392157,
            'LR' : 0.8698777777777774
            }
max_accuracy = 0 
max_recall = 0 
max_precision = 0
for i in accuracy_contador.values():
    if i > max_accuracy:
        max_accuracy = i 
for i in recall_contador.values():
    if i > max_recall:
        max_recall = i 
for i in precision_contador.values():
    if i > max_precision:
        max_precision = i 
modelo_max_accuracy = list(accuracy_contador.keys())[list(accuracy_contador.values()).index(max_accuracy)]
modelo_max_recall = list(recall_contador.keys())[list(recall_contador.values()).index(max_recall)]
modelo_max_precision = list(precision_contador.keys())[list(precision_contador.values()).index(max_precision)]

print(f"El modelo con mayor accuracy es {modelo_max_accuracy} con un accuracy de {max_accuracy}")
print(f"El modelo con mayor recall es {modelo_max_recall} con un recall de {max_recall}")
print(f"El modelo con mayor precision es {modelo_max_precision} con un precision de {max_precision}")

# Funciones medidoras de F1-Score (manual) 

def medidor_f1(recall, precision, nombre_modelo):
    f1 = 2 * ((recall * precision) / (recall + precision))

    print(f'{f1} de puntuación F1 para el modelo {nombre_modelo}')

medidor_f1(0.35344827586206895, 0.7068965517241379, 'KNN')
medidor_f1(0.75, 0.8130841121495327, 'DT')
medidor_f1(0.7327586206896551, 0.9659090909090909, 'RF')
medidor_f1(0.7586206896551724, 0.8627450980392157, 'SVM')
medidor_f1(0.7498484748389891, 0.8698777777777774, 'LR')

f1_contador = {'KNN': 0.47126436781609193,
               'DT' : 0.7802690582959642,
    'RF': 0.8333333333333334,
    'SVM': 0.8073394495412843,
    'LR': 0.8054157594954122
}
max_f1 = 0
for i in f1_contador.values():
    if i > max_f1:
        max_f1 = i
modelo_max_f1 = list(f1_contador.keys())[list(f1_contador.values()).index(max_f1)]

print(f"El modelo con mayor puntuación F1 es {modelo_max_f1} con un accuracy de {max_f1}")

# Validación Cruzada de los Modelos

from sklearn.model_selection import cross_val_score, KFold
def validacion_cruzada(modelo, nombre_modelo):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cross_val = cross_val_score(modelo, x, y, cv=kfold)

    print(f'Validación Cruzada a 5 Vueltas \n {cross_val}, {nombre_modelo}')

knn_cros_val = validacion_cruzada(knn2, 'KNN')
dt_cross_val = validacion_cruzada(dt2, 'DT')
rf_cross_val = validacion_cruzada(rf2, 'RF')
svm_cros_val = validacion_cruzada(svm2, 'SVM')
lr_cros_val = validacion_cruzada(lr2, 'LR')

validaciones = {'KNN': [0.69333333, 0.63666667, 0.68666667, 0.70333333, 0.73],
               'DT' : [0.82666667, 0.79333333, 0.82333333, 0.82, 0.80666667],
    'RF': [0.90666667, 0.86, 0.89333333, 0.90666667, 0.88],
    'SVM': [0.86, 0.86333333, 0.86, 0.86, 0.83],
    'LR': [0.86, 0.85333333, 0.85, 0.85666667, 0.83]
}
for nombre_modelo, puntuaciones in validaciones.items():
    print(f'El modelo {nombre_modelo} tiene una media de {np.mean(puntuaciones)}')
validaciones_score = {'KNN': 0.6900000000000001,
               'DT' : 0.8140000000000001,
    'RF': 0.889333334,
    'SVM': 0.854666666,
    'LR': 0.85
}
max_cross_val = 0
for i in validaciones_score.values():
    if i > max_cross_val:
        max_cross_val = i
        modelo_max_cross_val = list(validaciones_score.keys())[list(validaciones_score.values()).index(max_cross_val)]

print(f'El modelo con mejor media sobre la Validación Cruzada es {modelo_max_cross_val}, con una media de {max_cross_val}')
## Graficación de las Evaluaciones de los Modelos ##

# Graficación del Accuracy

plt.figure(figsize=(10, 5))
plt.bar(['KNN', 'DT', 'RF', 'SVM', 'LR'], accuracy_contador.values(), color = 'blue')
plt.xlabel('Modelos')
plt.ylabel('Accuracy')
plt.title('Accuracy de los Modelos')
plt.show()

# Graficación del Recall

plt.figure(figsize=(10, 5))
plt.bar(['KNN', 'DT', 'RF', 'SVM', 'LR'], recall_contador.values(), color = 'purple')
plt.xlabel('Modelos')
plt.ylabel('Recall')
plt.title('Recall de los Modelos')
plt.show()

# Graficación del Precision

plt.figure(figsize=(10, 5))
plt.bar(['KNN', 'DT', 'RF', 'SVM', 'LR'], precision_contador.values(), color = 'red')
plt.xlabel('Modelos')
plt.ylabel('Precision')
plt.title('Precision de los Modelos')
plt.show()

# Graficación del F1-Score

plt.figure(figsize=(10, 5))
plt.bar(['KNN', 'DT', 'RF', 'SVM', 'LR'], f1_contador.values(), color = 'orange')
plt.xlabel('Modelos')
plt.ylabel('F1-Score')
plt.title('F1-Score de los Modelos')
plt.show()

# Graficación del Cross-Validation

plt.figure(figsize=(10, 5))
plt.bar(['KNN', 'DT', 'RF', 'SVM', 'LR'], validaciones_score.values(), color = 'green')
plt.xlabel('Modelos')
plt.ylabel('Cross-Val')
plt.title('Vallidación Cruzada de los Modelos')
plt.show()

# Evaluación de apoyo para la toma de decisión del modelo más óptimo para el caso

from statistics import mode
mas_repetido = {'Accuracy' : 'RF',
               'Recall' : 'SVM',
               'Precision' : 'RF',
               'F1-Score' : 'RF',
               'Cross-Validation' : 'RF',
               }     

print(f'El modelo candidato por número de métricas de evaluación que lidera es: {mode(mas_repetido.values())}')

# Matrices de Confusión de los Modelos

from matplotlib import colormaps
print(list(colormaps))
from sklearn.metrics import confusion_matrix

matrizConf = confusion_matrix(y_test, y_pred_knn2)
plt.figure(figsize=(6,5))
sns.heatmap(matrizConf, annot=True, cmap="nipy_spectral_r")
plt.xlabel("Reales")
plt.ylabel("Predicciones")
plt.title("Matriz de Confusión Modelo KNN")
plt.show()

matrizConf = confusion_matrix(y_test, y_pred_dt2)
plt.figure(figsize=(6,5))
sns.heatmap(matrizConf, annot=True, cmap="summer_r")
plt.xlabel("Reales")
plt.ylabel("Predicciones")
plt.title("Matriz de Confusión Modelo DT")
plt.show()

matrizConf = confusion_matrix(y_test, y_pred_rf2)
plt.figure(figsize=(6,5))
sns.heatmap(matrizConf, annot=True, cmap="rocket_r")
plt.xlabel("Reales")
plt.ylabel("Predicciones")
plt.title("Matriz de Confusión Modelo RF")
plt.show()

matrizConf = confusion_matrix(y_test, y_pred_svm2)
plt.figure(figsize=(6,5))
sns.heatmap(matrizConf, annot=True, cmap="flare_r")
plt.xlabel("Reales")
plt.ylabel("Predicciones")
plt.title("Matriz de Confusión Modelo SVM")
plt.show()

matrizConf = confusion_matrix(y_test, y_pred_lr2)
plt.figure(figsize=(6,5))
sns.heatmap(matrizConf, annot=True, cmap="coolwarm_r")
plt.xlabel("Reales")
plt.ylabel("Predicciones")
plt.title("Matriz de Confusión Modelo LR")
plt.show()
