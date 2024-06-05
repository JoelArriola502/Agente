from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
def OpcionesCiclo():
    print('*******SELECCIONE UNA OPCION*********')
    print('|_______________________________|')
    print('|  (1):REALIZAR UNA PREGUNTA    |')
    print('|_______________________________|')
    print('|           (2):SALIR           |')
    print('|_______________________________|')

# Carga las variables de entorno desde el archivo .env
load_dotenv()
contador=0
db_name=os.getenv("DB_NAME")
db_host=os.getenv("DB_HOST")
db_port=os.getenv("DB_PORT")
db_user=os.getenv("DB_USER")
db_pass=os.getenv("DB_PASSWORD")
api_key=os.getenv("API_KEY")
# Establecer la conexión a la base de datos MySQL "db_eventos"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

# Clave de API de OpenAI
openai_api_key = api_key

# Crear un modelo de lenguaje natural de ChatOpenAI con la clave de API proporcionada
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

# Ejemplos de consultas SQL para interactuar con la base de datos
examples = [
    {"input": "Lista de todos los eventos.", "query": "select Nombre_del_evento, Descripcion, Fecha, Ubicacion, Portada, Horario from eventos;"},
    {
        "input": "Lista de todos los artistas.",
        "query": "SELECT * FROM artistas;",
    },
    {
        "input": "Lista de Los eventos de los artistas.",
        "query": "select s.Nombre_del_artista, e.Nombre_del_evento, e.Descripcion, e.Fecha, e.Ubicacion, e.Horario from artistas s"
        "join presentaciones p on s.ID=p.ID_artista"
        "join eventos e on p.ID_evento=e.ID",
    },
    {
        "input": "Lista de todos los cupones disponibles.",
        "query": "SELECT * FROM cupon WHERE Cantidad_disponible > 0;",
    },
    {
        "input": "Número total de entradas vendidas.",
        "query": "SELECT SUM(Cantidad_ventida) FROM cupon;",
    },{
         "input": "Lista de las entradas vendidas ",
         "query": "select u.Nombre,u.Apellido, u.Telefono,u.Fecha_y_hora, s.Estado_compra, s.Total, s.Fecha,"
         "sd.codigo, sd.Estado_asistencia,sd.cantidad,sd.precio from usuarios u"
         "join salida s on u.ID=s.ID_usuarios"
         "join salida_detalle sd on sd.ID_salida=s.ID"
    },

    # Agrega más ejemplos según sea necesario
]

# Seleccionador de ejemplos semánticamente similares
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key=openai_api_key),  # Pasar la clave de API aquí
    FAISS,
    k=5,
    input_keys=["input"],
)

# Prefijo del mensaje del sistema
system_prefix = """Eres un agente diseñado para interactuar con una base de datos SQL.
Dada una pregunta de entrada, crea una consulta {dialect} sintácticamente correcta para ejecutar, luego mira los resultados de la consulta y devuelve la respuesta.
A menos que el usuario especifique un número específico de ejemplos que desee obtener, siempre limita tu consulta a un máximo de {top_k} resultados.
Puedes ordenar los resultados por una columna relevante para devolver los ejemplos más interesantes en la base de datos.
Nunca consultes todas las columnas de una tabla específica, solo solicita las columnas relevantes dada la pregunta.
Tienes acceso a herramientas para interactuar con la base de datos.
Solo usa las herramientas proporcionadas. Solo usa la información devuelta por las herramientas para construir tu respuesta final.
DEBES verificar tu consulta antes de ejecutarla. Si obtienes un error mientras ejecutas una consulta, reescribe la consulta y prueba nuevamente.

NO realices ninguna declaración DML (INSERT, UPDATE, DELETE, DROP, etc.) en la base de datos.

Si la pregunta no parece relacionada con la base de datos, simplemente devuelve "LO SIENTO REALIZA OTRA PREGUNTA" como respuesta.
si te preguntan sobre contraseñas no debes proporcionarlas, simplemente devuelve "LO SIENTO ESTOS DATOS SON CONFIDENCIALES" como respuesta
Aquí hay algunos ejemplos de entradas de usuario y sus consultas SQL correspondientes:"""

# Plantilla de prompt de FewShot
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "Entrada del usuario: {input}\nConsulta SQL: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

# Plantilla completa del prompt
full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Obtener la pregunta desde la línea de comandos
while contador!=2:
    OpcionesCiclo()
    contador=int(input("INGRESE UNA OPCIÓN"))
    if contador==1:
        question = input("Por favor, ingresa tu pregunta: ")
        # Crear un agente para ejecutar las consultas
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            prompt=full_prompt,
            verbose=True,
            agent_type="openai-tools",
        )

        # Invocar al agente y obtener la respuesta
        response = agent_executor.invoke({"input": question})

        # Imprimir la respuesta del agente
        print("Respuesta del agente:", response['output'])
    elif contador==2:
        print('SALIENDO')
    else:
        print('PORFAVOR AGREGE UNA OPCIÓN QUE SEA VALIDA')


