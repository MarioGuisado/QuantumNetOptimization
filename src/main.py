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


graph = nx.Graph()
graph.add_node(0)
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_node(4)
graph.add_node(5)
graph.add_node(6)
graph.add_node(7)
graph.add_node(8)

graph.add_edge(0, 1)
graph.add_edge(1, 0)
#graph.add_edge(0, 2)
graph.add_edge(1, 5)
graph.add_edge(5, 1)
graph.add_edge(2, 3)
graph.add_edge(3, 2)
graph.add_edge(3, 4)
graph.add_edge(3 ,5)
graph.add_edge(5, 3)
graph.add_edge(4 ,6)
graph.add_edge(6, 4)
graph.add_edge(5 ,8)
graph.add_edge(8 ,5)
graph.add_edge(4 ,7)
graph.add_edge(5 ,7)
graph.add_edge(7 ,4)
graph.add_edge(7, 8)
graph.add_edge(8 ,7)
graph.add_edge(6 ,8)
graph.add_edge(8 ,6)



N = len(graph.nodes())
E = len(graph.edges())


initializer = Initializer(graph)
#initializer.draw()

builder = QUBObuilder()
functions = {}
#functions[1] = [{2}]
functions[8] = [{9}]
functions[6] = [{8}]
QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,variable_constrain = builder.get_QUBO_model(graph, 0, 2, functions)
solver = QUBOSolverCPU(
    number_iterations=200000,
    number_runs=10,
    scaling_bit_precision=32,
    auto_tuning=AutoTuning.AUTO_SCALING_AND_SAMPLING)

for p in cost_function, first_constrain :
    solution_list = solver.minimize(p)
    configuration = solution_list.min_solution.configuration
    print("Min %s: at %s value %f" % (p, configuration, p.compute(configuration)) )


solution_list = solver.minimize(QUBOexpression)
my_bit_array = solution_list.min_solution.extract_bit_array('x')
my_bit_array.draw(axis_names=['i', 'j'])