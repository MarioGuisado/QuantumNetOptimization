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

    def draw(self):
        nx.draw(self.graph, with_labels=True, node_color='lightblue', node_size=1000, font_size=12, font_weight='bold', arrows=True)
        plt.show()