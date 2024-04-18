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
graph.add_nodes_from([0, 1, 2, 3, 4, 5])



graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(2, 4)
graph.add_edge(4, 3)
graph.add_edge(1, 3)
graph.add_edge(3, 4)
graph.add_edge(3, 5)
graph.add_edge(4, 2)

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


initializer = Initializer(graph)
initializer.draw()

builder = QUBObuilder()
functions = {}
functions[2] = [{2}]
#functions[1] = [{1}]
QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,variable_constrain = builder.get_QUBO_model(graph, 0, 5, functions)
solver = QUBOSolverCPU(
    number_iterations=20000,
    number_runs=10,
    scaling_bit_precision=32,
    auto_tuning=AutoTuning.AUTO_SCALING_AND_SAMPLING)

solution_list = solver.minimize(QUBOexpression)
for p in cost_function, first_constrain, second_constrain ,third_constrain ,variable_constrain :
    configuration = solution_list.min_solution.configuration
    print("Min %s: at %s value %f" % (p, configuration, p.compute(configuration)) )


solution_list = solver.minimize(QUBOexpression)
my_bit_array = solution_list.min_solution.extract_bit_array('x')
my_bit_array.draw(axis_names=['i', 'j'])