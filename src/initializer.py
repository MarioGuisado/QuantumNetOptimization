from IPython.display import display, HTML
from random import uniform, seed
from tabulate import tabulate
from numpy import argmax
import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from random import random

class Initializer:
    def __init__(self, graph):
        self.graph = graph
        self.nodes_with_functions = {}
        for node in self.graph.nodes():
            try:
                function = self.graph.nodes[node]["function"]
                print(f"El nodo {node} tiene la función {function}")
                if node in self.nodes_with_functions:
                    self.nodes_with_functions[node].append(function)
                else:
                    self.nodes_with_functions[node] = [function]
            except KeyError:
               print(f"El nodo {node} no tiene una función asociada")
        

    def draw(self):
        colors = ['g']*len(self.graph)
        pos = nx.spring_layout(self.graph)
        fig = plt.figure(num='Graph (created at %s)' % datetime.now(), figsize=(9.0, 6.0))
        nx.draw(self.graph, pos=pos, with_labels=True,
                nodelist=range(len(colors)), node_color=colors)
        plt.show()

    def get_graph(self):
        return self.graph
    
    def get_nodes_with_functions(self):
        return self.nodes_with_functions