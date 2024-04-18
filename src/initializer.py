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
        
        print(self.nodes_with_functions)
        

    def draw(self):
        nx.draw(self.graph, with_labels=True, node_color='lightblue', node_size=1000, font_size=12, font_weight='bold', arrows=True)
        plt.show()

    def get_graph(self):
        return self.graph
    
    def get_nodes_with_functions(self):
        return self.nodes_with_functions