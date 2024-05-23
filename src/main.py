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

connections = read_connections('./instancias/6 nodos/topologia_6.DAT')
#print(connections)
#coste_conexion = 2
#connections = np.floor_divide(connections, coste_conexion)

#print(connections)

for i, row in enumerate(connections):
    for j, value in enumerate(row):
        if value != 0:
            graph.add_edge(i, j)

initializer = Initializer(graph)
initializer.draw()

builder = QUBObuilder()
functions = {}
#functions[2] = [{3,5}]
#functions[5] = [{5}]

# Abre el archivo en modo de lectura
with open('./instancias/6 nodos/nodos_recursos_6.DAT', 'r') as file:
    # Lee la línea del archivo y la divide en números
    resources = file.readline().split()

# Convierte los números a enteros
resources = [int(resource) for resource in resources]



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
QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,fourth_constrain,fifth_constrain, sixth_constrain, variable_constrain = builder.get_QUBO_model(graph, 0, 4, functions, connections,resources, 1, 2, 2, 2, 2, 2)
solver = QUBOSolverCPU(
number_iterations=1500000,
number_runs=50,
scaling_bit_precision=16,
auto_tuning=AutoTuning.AUTO_SCALING_AND_SAMPLING)

solution_list = solver.minimize(QUBOexpression)
configuration = solution_list.min_solution.configuration

for p in cost_function, first_constrain, second_constrain ,third_constrain , fourth_constrain,fifth_constrain, sixth_constrain, variable_constrain :
    print("Min %s: at %s value %f" % (p, configuration, p.compute(configuration)))


solution_list = solver.minimize(QUBOexpression)
my_bit_array = solution_list.min_solution.extract_bit_array('x')
my_bit_array.draw(axis_names=['i', 'j', 'a'])