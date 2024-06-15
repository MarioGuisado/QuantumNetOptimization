from IPython.display import display, HTML
from random import uniform, seed
from tabulate import tabulate
from numpy import argmax
import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from random import random
from initializer import * 
from QUBObuilder import *
from dimod.generators import and_gate
from dwave.system import LeapHybridSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

graph = nx.DiGraph()

N = len(graph.nodes())
E = len(graph.edges())



def read_connections(path):
    matrix = []
    with open(path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.split()]
            matrix.append(row)
    return matrix

# Define los nombres de los archivos
connection_file_names = {
    '1': './instancias/6 nodos/topologia_6.DAT',
    '2': './instancias/19 nodos/topologia_19.DAT',
    '3': './instancias/15 nodos/topologia_15.DAT',
}

function_file_names = {
    '1': './instancias/6 nodos/nodos_6.DAT',
    '2': './instancias/19 nodos/nodos_19.DAT',
    '3': './instancias/15 nodos/nodos_15.DAT',
}

resources_file_names = {
    '1': './instancias/6 nodos/nodos_recursos_6.DAT',
    '2': './instancias/19 nodos/nodos_recursos_19.DAT',
    '3': './instancias/15 nodos/nodos_recursos_15.DAT',
}

connection_file_name = None
function_file_name = None
resources_file_name = None


# Solicita al usuario que introduzca el número del archivo hasta que se introduzca un número válido
while connection_file_name is None:
    file_num = input("Please enter the connection file number (1, 2, or 3): ")
    if file_num == '1':
        connection_file_name = connection_file_names['1']
        function_file_name = function_file_names['1']
        resources_file_name = resources_file_names['1']
        hybrid_time = 10
        dot_size = 200
        alpha1 = 50
        alpha2 = 50
        alpha3 = 50
        variable_alpha = 50
        alpha4 = 50
        alpha5 = 50
        alpha6 = 50
        alpha7 = 50
        alpha8 = 1
    elif file_num == '2':
        connection_file_name = connection_file_names['2']
        function_file_name = function_file_names['2']
        resources_file_name = resources_file_names['2']
        hybrid_time = 40
        dot_size = 100
        alpha1 = 5000
        alpha2 = 5000
        alpha3 = 5000
        variable_alpha = 3000
        alpha4 = 5000
        alpha5 = 3000
        alpha6 = 3000
        alpha7 = 3000
        alpha8 = 10
    elif file_num == '3':
        connection_file_name = connection_file_names['3']
        function_file_name = function_file_names['3']
        resources_file_name = resources_file_names['3']
        hybrid_time = 25
        dot_size = 100
        alpha1 = 2000
        alpha2 = 2000
        alpha3 = 5000
        variable_alpha = 2000
        alpha4 = 500
        alpha5 = 500
        alpha6 = 500
        alpha7 = 500
        alpha8 = 5
    elif connection_file_name is None:
        print("Invalid file number. Please try again.")

connections = read_connections(connection_file_name)


availableConnections = 0
for sublist in connections:
    for i in sublist:
        if i != 0:
            availableConnections += 1


coste_conexion = 2
connections = np.floor_divide(connections, coste_conexion)
print(connections)


for i, row in enumerate(connections):
    for j, value in enumerate(row):
        if value != 0:
            graph.add_edge(i, j)

initializer = Initializer(graph)
initializer.draw()

builder = QUBObuilder()
functions = {}
desired_functions = {}


# Solicita al usuario que introduzca los números separados por comas
input_str = input("Please enter the desired functions, separated by spaces: ")

# Divide la cadena en una lista de cadenas
input_list = input_str.split(' ')

# Convierte la lista de cadenas en un conjunto de enteros
desired_functions = {int(num_str) for num_str in input_list}

print(desired_functions)


# Abre el fichero
with open(function_file_name, 'r') as f:
    # Lee cada línea del fichero
    nodo = 0
    for line in f:
        # Divide la línea en nodos y funciones
        funciones = list(map(int, line.split()))
        for i in range(len(funciones)):
            if funciones[i] in desired_functions:
                if nodo not in functions:
                    functions[nodo] = set()
                functions[nodo].add(funciones[i])
        nodo += 1
       
print("hola", functions)
# Abre el archivo en modo de lectura
with open(resources_file_name, 'r') as file:
    # Lee la línea del archivo y la divide en números
    resources = file.readline().split()

# Convierte los números a enteros
resources = [int(resource) for resource in resources]

print("Los recursos son: ", resources)


num_agents = -1

# Solicita al usuario que introduzca el número de agentes hasta que se introduzca un número válido
while num_agents < 1 or num_agents > 3:
    try:
        num_agents = int(input("Please enter the number of agents (between 1 and 3): "))
        if num_agents < 1 or num_agents > 3:
            print("Invalid number of agents. Please enter a number between 1 and 3.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,fourth_constrain,fifth_constrain, sixth_constrain,seventh_constrain, variable_constrain = builder.get_QUBO_model(graph, 1, 10, functions, connections,resources,num_agents,alpha1, alpha2, alpha3, variable_alpha, alpha4, alpha5, alpha6, alpha7, alpha8)
solver = QUBOSolverCPU(
number_iterations=200000,
number_runs=30,
scaling_bit_precision=16,
auto_tuning=AutoTuning.AUTO_SCALING_AND_SAMPLING)

#solution_list = solver.minimize(QUBOexpression)
#configuration = solution_list.min_solution.configuration

#for p in cost_function, first_constrain, second_constrain ,third_constrain , fourth_constrain,fifth_constrain, sixth_constrain,seventh_constrain, variable_constrain :
#    print("Min %s: at %s value %f" % (p, configuration, p.compute(configuration)))


#solution_list = solver.minimize(QUBOexpression)
#my_bit_array = solution_list.min_solution.extract_bit_array('x')
#my_bit_array.draw(axis_names=['i', 'j', 'a'])


def as_bqm(self) -> 'dimod.BinaryQuadraticModel':
    """
    The polynomial is returned as a :class:`dimod.BinaryQuadraticModel` object.

    :return: qubo as dimod.BinaryQuadraticModel object
    :rtype: dimod.BinaryQuadraticModel
    """

    try:
        import dimod
    except Exception as oops:
        print('\n\n' + (100 * '#'))
        print('pip install dwave-ocean-sdk')
        print((100 * '#') + '\n\n')
        raise oops

    return dimod.BinaryQuadraticModel(
        {i0: self._p1[i0] for i0 in self._p1},
        {(i0, i1): self._p2[i0][i1] for i0 in self._p2 for i1 in self._p2[i0]},
        self._p0,
        dimod.BINARY)
        

bqm_problem=QUBOexpression.as_bqm()
os.environ['DWAVE_API_TOKEN']='DEV-937f902268d58e5ac473ef59a35c57b0027979c4'

sampler = LeapHybridSampler()

answer = sampler.sample(bqm_problem, time_limit=hybrid_time, label='Hybrid Exec')
print(answer)
samples = []
for datum in answer.data(['sample', 'energy']):
   print(datum.sample, datum.energy)
   first_N_bits = {k: v for k, v in datum.sample.items() if k < availableConnections * num_agents}
   samples.append(first_N_bits)
   break

print(samples)
#solution_dwave=list(datum.sample.values())

#print( "QUBOexpression  = %10.6f" % (QUBOexpression.compute(solution_dwave)) )


#sample_time = time.time()

# Set a QPU sampler
#sampler = EmbeddingComposite(DWaveSampler())

#num_reads = 4200
#sampleset = sampler.sample(bqm_problem, num_reads=num_reads, label='Purely Quantum Exec')

#samples = []
#for datum in sampleset.lowest().data(['sample', 'energy']):
#    print(datum.sample, datum.energy)
#    first_N_bits = {k: v for k, v in datum.sample.items() if k < availableConnections * num_agents}
#    samples.append(first_N_bits)
#    break

#print(samples)

# Itera sobre cada diccionario en samples
# Calcula el número de filas y columnas para los subplots
# Calcula el número de filas y columnas para los subplots
n = len(samples) * num_agents  # Ajusta n para tener en cuenta los agentes
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))

# Crea una figura
fig = plt.figure(figsize=(cols*6, rows*6))

for i, sample in enumerate(samples):
    for agent in range(num_agents):
        # Crea una matriz basada en la matriz de conexiones
        matrix = np.zeros_like(connections)
        contador = 0
        for j, value in np.ndenumerate(connections):
            if value == 0:
                matrix[j] = 0
            else:
                # Usa el número de agente y el contador para calcular el índice correcto
                matrix[j] = sample[contador*num_agents+agent]
                contador += 1

        # Selecciona el subplot
        ax = fig.add_subplot(rows, cols, i*num_agents+agent+1)

        # Dibuja la matriz como una serie de círculos
        for (x, y), value in np.ndenumerate(matrix):
            if connections[x][y] == 0:
                color = 'lightblue'
            elif value == 0:
                color = 'blue' 
            else:
                color = 'red'
            ax.scatter(x, y, c=color, s=dot_size) 

        # Establece las ubicaciones y etiquetas de las marcas de los ejes
        ax.set_xticks(np.arange(matrix.shape[1]), np.arange(1, matrix.shape[1] + 1))
        ax.set_yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))

        # Agrega etiquetas a los ejes
        ax.set_xlabel('i')
        ax.set_ylabel('j')

        # Establece el título del subplot
        ax.set_title(f'Sample {i+1}, Agent {agent+1}')

# Muestra la figura
plt.show()