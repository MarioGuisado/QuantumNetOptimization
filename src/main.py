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

#graph.add_edge(5, 6)
#graph.add_edge(6, 3)
#graph.add_edge(3, 6)
#graph.add_edge(6, 7)
#graph.add_edge(7, 8)
#graph.add_edge(8, 2)
#graph.add_edge(2, 8)
#graph.add_edge(8, 7)



N = len(graph.nodes())
E = len(graph.edges())



def read_connections(path):
    matrix = []
    with open(path, 'r') as file:
        for line in file:
            row = [int(x) for x in line.split()]
            matrix.append(row)
    return matrix

#connections = read_connections('./instancias/6 nodos/topologia_6.DAT')
connections = read_connections('./instancias/prueba/prueba.DAT')


availableConnections = 0
for sublist in connections:
    for i in sublist:
        if i != 0:
            availableConnections += 1

#print(availableConnections)

#print(connections)
#coste_conexion = 2
#connections = np.floor_divide(connections, coste_conexion)

#print(connections)

#for i, row in enumerate(connections):
#    for j, value in enumerate(row):
#        if value != 0:
#            graph.add_edge(i, j)

graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(0, 3)
graph.add_edge(3, 2)
initializer = Initializer(graph)
initializer.draw()

builder = QUBObuilder()
functions = {}
#functions[3] = [{3,5}]
#functions[5] = [{5}]

# Abre el archivo en modo de lectura
with open('./instancias/6 nodos/nodos_recursos_6.DAT', 'r') as file:
    # Lee la línea del archivo y la divide en números
    resources = file.readline().split()

# Convierte los números a enteros
#resources = [int(resource) for resource in resources]
resources = [10, 3, 10, 0]



#with open('./instancias/6 nodos/nodos_6.DAT', 'r') as file:
    #for index, line in enumerate(file):
        # Dividir la línea en elementos y convertirlos en enteros
        #elementos = list(map(int, line.split()))
        
        # Filtrar los ceros y crear un conjunto de funciones
        #funciones = set(filter(lambda x: x != 0, elementos))
        
        # Almacenar el conjunto en el diccionario
        #functions[index] = [funciones]

#print(functions)


#alpha1 = 10 * N
#alpha2 = 10 * N
#alpha3 = 1000* N
#alpha4 = 1000 * N
#alpha5 = 100 * N
QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,fourth_constrain,fifth_constrain, sixth_constrain,seventh_constrain, variable_constrain = builder.get_QUBO_model(graph, 0, 2, functions, connections,resources, 1, 2, 2, 2, 2, 2)
solver = QUBOSolverCPU(
number_iterations=200000,
number_runs=20,
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
os.environ['DWAVE_API_TOKEN']='DEV-6d3884fc26ae2d49987a7b350237d126ec957ad7'

#sampler = LeapHybridSampler()
#answer = sampler.sample(bqm_problem)
#print(answer)

#for datum in answer.data(['sample', 'energy']):
#   print(datum.sample, datum.energy)

#solution_dwave=list(datum.sample.values())

#print( "QUBOexpression  = %10.6f" % (QUBOexpression.compute(solution_dwave)) )


sample_time = time.time()

# Set a QPU sampler
sampler = EmbeddingComposite(DWaveSampler())

num_reads = 2000
#sampleset = sampler.sample(bqm_problem, num_reads=num_reads, label='Purely Quantum Exec')

samples = []
contador = 0
num_agents = 2
#for datum in sampleset.lowest().data(['sample', 'energy']):
#    if contador < 1:
#        print(datum.sample, datum.energy)
#        first_N_bits = {k: v for k, v in datum.sample.items() if k < availableConnections * num_agents}
#        samples.append(first_N_bits)
#    contador += 1

samples = [{0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}]
print(samples)

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
            ax.scatter(x, y, c=color, s=200) 

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