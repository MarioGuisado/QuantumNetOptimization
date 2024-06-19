#Author: Mario Guisado García
#Github: MarioGuisado

import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from constants import connection_file_names, function_file_names, resources_file_names

class Initializer:
    def __init__(self, graph):
        self.graph = graph
        self.agents_limit = 1

    def draw(self):
        labels = {node: node + 1 for node in self.graph.nodes()}
        nx.draw(self.graph, with_labels=True, labels=labels, node_color='lightblue', node_size=1000, font_size=12, font_weight='bold', arrows=True)
        plt.show()

    def read_connections(self, path):
        matrix = []
        with open(path, 'r') as file:
            for line in file:
                row = [int(x) for x in line.split()]
                matrix.append(row)
        return matrix
    
    def process_connections(self, connections):
        available_connections = 0
        for sublist in connections:
            for i in sublist:
                if i != 0:
                    available_connections += 1

        connection_cost = 2
        connections = np.floor_divide(connections, connection_cost)

        return available_connections, connections

    def get_start_end_nodes(self):
        start_node = int(input("Please enter the start node (starting at 1): "))
        end_node = int(input("Please enter the end node (ending at N): "))
        return start_node - 1, end_node - 1
    
    def get_file_data(self):
        connection_file_name = None
        while connection_file_name is None:
            file_num = input("Please select the graph (6 nodes -> 1 | 19 nodes -> 2): ")
            if file_num == '1':
                connection_file_name = connection_file_names['1']
                function_file_name = function_file_names['1']
                resources_file_name = resources_file_names['1']
                hybrid_time = 10
                dot_size = 200
                alpha_cost_function = 1
                self.agents_limit = 3
            elif file_num == '2':
                connection_file_name = connection_file_names['2']
                function_file_name = function_file_names['2']
                resources_file_name = resources_file_names['2']
                hybrid_time = 35
                dot_size = 100
                alpha_cost_function = 5
                self.agents_limit = 3
            elif connection_file_name is None:
                print("Invalid file number. Please try again.")

        connections = self.read_connections(connection_file_name)
        return connections, function_file_name, resources_file_name, hybrid_time, dot_size, alpha_cost_function
    
    def get_num_agents(self):
        num_agents = -1
        while num_agents < 1 or num_agents > self.agents_limit:
            try:
                if self.agents_limit == 3:
                    num_agents = int(input("Please enter the number of agents (between 1 and 3): "))
                else:
                    num_agents = int(input("Please enter the number of agents (between 1 and 2): "))
                if num_agents < 1 or num_agents > self.agents_limit:
                    print("Invalid number of agents.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        return num_agents