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

        var_shape_set = VarShapeSet(BitArrayShape(name='x', shape=(N, N),axis_names=['i', 'j'], constant_bits=problem_constant_bits))
        
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
    

 
        nodes_with_functions = functions.keys()
        
        function_values = functions.values()
        functions_list = [] 
        for values in function_values:
            for value in values:
                if type(value) == set:
                    for k in value:
                        functions_list.append(k)
                else:
                    functions_list.append(value)
        functions_list.sort()
        #print(functions_list)
        
        N = 4
        minimum_bits = math.ceil(math.log(N+1, 2))
        print("El número mínimo de bits es: ", minimum_bits)
        print("Podemos representar hasta el: ", 2 ** minimum_bits - 1)
        print("Solo necesitamos representar hasta el: ", N)
        ultimo_coef = 2**(minimum_bits - 1)
        resta_a_coeficiente = 2 ** minimum_bits - N - 1
        print("El ultimo coeficiente es: ", ultimo_coef)
        print("Debo restarle a este coeficiente: ", resta_a_coeficiente)

       
        var_slack = VarSlack(name='slack_variable',start=0,step=1,stop=N,slack_type=SlackType.binary)
        var_shape = VarShapeSet(var_slack)
        variable_constrain = BinPol(var_shape_set)
        for function in functions_list:  
            variable_constrain_aux = BinPol(var_shape_set)
            for i in range(N):
                for j in nodes_with_functions:
                    variable_constrain_aux.add_term(1,("x",i,j))
                variable_constrain_aux.add_term(1,("x",initial_node,i))
                variable_constrain = variable_constrain_aux - BinPol(var_shape).add_slack_variable('slack_variable')
                variable_constrain_aux.power(2)
            variable_constrain = variable_constrain + variable_constrain_aux
        
        print(variable_constrain)

            

                
                

            
        

    




