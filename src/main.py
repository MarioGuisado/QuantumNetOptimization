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

for i, row in enumerate(connections):
    for j, value in enumerate(row):
        if value != 0:
            graph.add_edge(i, j)

initializer = Initializer(graph)
initializer.draw()

builder = QUBObuilder()
functions = {}
functions[3] = [{3}]

#alpha1 = 10 * N
#alpha2 = 10 * N
#alpha3 = 1000* N
#alpha4 = 1000 * N
#alpha5 = 100 * N
QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,fourth_constrain,fifth_constrain, sixth_constrain, variable_constrain = builder.get_QUBO_model(graph, 0, 4, functions, connections, 1, 2, 2, 2, 2, 2)
solver = QUBOSolverCPU(
number_iterations=200000,
number_runs=20,
scaling_bit_precision=32,
auto_tuning=AutoTuning.AUTO_SCALING_AND_SAMPLING)

solution_list = solver.minimize(QUBOexpression)
configuration = solution_list.min_solution.configuration

for p in cost_function, first_constrain, second_constrain ,third_constrain , fourth_constrain,fifth_constrain, sixth_constrain, variable_constrain :
    print("Min %s: at %s value %f" % (p, configuration, p.compute(configuration)))


solution_list = solver.minimize(QUBOexpression)
my_bit_array = solution_list.min_solution.extract_bit_array('x')
my_bit_array.draw(axis_names=['i', 'j', 'a'])