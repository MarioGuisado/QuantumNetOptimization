from IPython.display import display, HTML
from random import uniform, seed
from tabulate import tabulate
from numpy import argmax
import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from random import random

class QUBObuilder:
    def get_QUBO_model(self, graph, initial_node, final_node, functions):
        N = len(graph.nodes())
        
        problem_constant_bits = np.full((N,N), -1, np.int8)

        for i in range(N):
            for j in range(N):
                if not graph.has_edge(i,j) or i == j:
                    problem_constant_bits[i][j] = 0
        
        print(problem_constant_bits)
        
        function_values = functions.values()
        functions_list = [] 
        for values in function_values:
            for value in values:
                if type(value) == set:
                    for k in value:
                        if k not in functions_list:
                            functions_list.append(k)
                else:
                    if value not in functions_list:
                        functions_list.append(value)
        print(functions_list)
        
        initial_node_functions = []
        if initial_node in functions:
            for function in functions[initial_node]:
                for values in function:
                    initial_node_functions.append(values)
        
        print(initial_node_functions)

        lista_slack = []
        for i in functions_list:    
            if i in initial_node_functions:
                lista_slack.append(VarSlack(name='slack_variable_'+str(i),start=0,step=1,stop=N-1,slack_type=SlackType.binary))
            else:
                lista_slack.append(VarSlack(name='slack_variable_'+str(i),start=0,step=1,stop=N-2,slack_type=SlackType.binary))

        var_shape_set = VarShapeSet(BitArrayShape(name='x', shape=(N, N),axis_names=['i', 'j'], constant_bits=problem_constant_bits), *lista_slack)
        
        cost_function = BinPol(var_shape_set)
        for i in range(N):
            for j in range(N):
                cost_function.add_term(1,("x",i,j))


        first_constrain = BinPol(var_shape_set)
        first_constrain.set_term(1,())

        for j in range(N):
            first_constrain.add_term(-1,("x",initial_node,j))
        first_constrain.power(2)
        
        second_constrain = BinPol(var_shape_set)
        second_constrain.set_term(1,())
        for i in range(N):
            second_constrain.add_term(-1,("x",i,final_node))
        second_constrain.power(2)

        third_constrain = BinPol(var_shape_set)
        for k in range(N):
            if k != initial_node and k != final_node:
                third_constrain_aux = BinPol(var_shape_set)

                for i in range(N):
                    third_constrain_aux.add_term(1,("x",i,k))
                for j in range(N):
                    third_constrain_aux.add_term(-1,("x",k,j))
                third_constrain_aux.power(2)
                third_constrain = third_constrain + third_constrain_aux
    
        variable_constrain = BinPol(var_shape_set)
        for function in functions_list:        
            variable_constrain_aux = BinPol(var_shape_set)
            variable_constrain_aux.set_term(-1,()) 
            nodes_with_functions = []
            for node in functions:
                for group in functions[node]:
                    if function in group:
                        nodes_with_functions.append(node)
            for i in range(N):
                for j in nodes_with_functions: 
                    variable_constrain_aux.add_term(1,("x",i,j))
            print(nodes_with_functions)
            if function in initial_node_functions:
                variable_constrain_aux.add_term(1,()) 
            variable_constrain_aux.add_slack_variable('slack_variable_'+str(function), factor=-1)
            variable_constrain_aux.power(2)
            variable_constrain = variable_constrain + variable_constrain_aux
        
        print(variable_constrain)
        
        alpha1 = 500 * N
        alpha2 = 500 * N
        alpha3 = 500 * N
        alpha4 = 1000 * N
        QUBOexpression =  cost_function + alpha1*first_constrain + alpha2*second_constrain + alpha3*third_constrain + alpha4*variable_constrain
        
        return QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,variable_constrain

            

                
                

            
        

    




