import pandas as pd
import numpy as np
import copy 




#EXTRACT 
data = pd.read_csv("data_socio_demografica.csv")
NSE = pd.read_csv("nse_inegi.csv", sep=";");




#TRANSFORM 

#Eliminar columna ID
data = data.drop(columns=["ID"]);

#Crear una copia para manipular los datos
datos_prueba = copy.deepcopy(data.head());

#Con el dataframe se tiene no limite en el nivel socio economico mas alto, que es de 100,000 en adelate por lo que para no tener problemas a la hora de trbajar con los int se asigna np.inf
NSE['Max'] = NSE['Max'].fillna(np.inf)

#Se anade la columna ingrso mensual para el calculo del nivel socioeconomico 
data["Ingreso mensual"] = (data["Ingreso anual"]/12).astype(int);

#iterador para asignar el nivel socioeconomico segun su ingreso
def clasificar_nivel(ingreso):
    for _, fila in NSE.iterrows():
        if fila['Min'] <= ingreso <= fila['Max']:
            return fila['Nivel Socioeconómico'], fila['Descripcion']
    return 'Desconocido', 'Sin descripción'

data[['Nivel socioeconomico', 'Descripción NSE']] = data['Ingreso mensual'].apply(clasificar_nivel).apply(pd.Series)

#dar valores null a toda una fila para verificar el funcionamiento del tratamiento de valores nulos
# data.loc[1] = np.nan
# print(data);

#definir si en las columnas hay filas vacias
for col in data.columns:
    if data[f"{col}"].isnull().sum() > 0:
        #si existen valores nulos en la columna se rellena con el promedio con con la moda 
        if data[f"{col}"].dtype == 'object':
            #obtener el valor mas comun en la columna sexo, si existen 2 valores con la misma cantidad o mas se selecciona el primero con [0]
            data[f"{col}"] = data[f"{col}"].fillna(data[f"{col}"].mode()[0])
        else:
            #obtener la media y asignarla al valor nulo
            data[f"{col}"] = data[f"{col}"].fillna(data[f"{col}"].mean())


# Diccionarios de codificación
CODIFICACION = {
    'Sexo': {
        'Mujer': 0,            
        'Hombre': 1
    },
    
    'Estado civil': {
        'soltero/a': 0,         
        'casado/a': 1,          
        'divorciado/a': 2,      
        'viudo/a': 3,           
        'unión libre': 4      
    },
    
    'Educacion': {
        'primaria': 0,                
        'secundaria': 1,              
        'preparatoria': 2,           
        'universidad': 3
    },
    
    'Ocupacion': {
        'desempleado/a': 0,    
        'desempleado/a / no calificado/a': 1,       
        'obrero/a no calificado/a': 2, 
        'empleado/a informal': 3,      
        'empleado/a calificado/a / funcionario/a': 4,  
        'dueño/a de medios': 5      
    },
    
    'Tamano de asentamiento': {
        'Rural': 0,                   
        'Ciudad mediana': 1, 
        'Metrópoli': 2         
    },
    
    'Nivel socioeconomico':{
        'A/B': 0,
        'C+' : 1,
        'C' : 2, 
        'C-': 3,
        'D+': 4,
        'D': 5,
        'E': 6
    }
}



def codificar_datos(dataframe, columnas, codificacion):
    df = dataframe.copy()
    for columna in columnas:
        if columna in df.columns and columna in codificacion:
            df[columna] = df[columna].map(codificacion[columna]).astype("Int64")
    return df

columnas_codificar = ['Sexo', 'Estado civil', 'Educacion', 'Ocupacion', 'Nivel socioeconomico'];

data2 = codificar_datos(data, columnas_codificar,CODIFICACION);

print(data2.sample(5));

#revisar coreciones de el diccionario con el respectivo indice 
indices = data2[data2.isnull().any(axis=1)].index
print(indices)
print(data.iloc[indices])


print(data2.isnull().sum());

#LOAD

#para el caso practico los datos soslamente se pasaran localmente en un cvs
data2.to_csv("data_ml_ETL_sd.csv", index=False);