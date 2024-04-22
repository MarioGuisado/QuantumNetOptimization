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
    def get_QUBO_model(self, graph, initial_node, final_node, functions, alpha1, alpha2, alpha3, alpha4, alpha5):
        N = len(graph.nodes())
        
        problem_constant_bits = np.full((N,N), -1, np.int8)

        for i in range(N):
            for j in range(N):
                if not graph.has_edge(i,j) or i == j:
                    problem_constant_bits[i][j] = 0
        
        #print(problem_constant_bits)
        
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
        #print(functions_list)
        
        initial_node_functions = []
        if initial_node in functions:
            for function in functions[initial_node]:
                for values in function:
                    initial_node_functions.append(values)
        
        #print(initial_node_functions)

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
                #print("Añadiendo termino: ", 1,("x",i,j) ," a la funcion de costo")
                cost_function.add_term(1,("x",i,j))
                #print(cost_function)

        #print("La funcion de costo es: ", cost_function)


        first_constrain = BinPol(var_shape_set)
        first_constrain.set_term(1,())

        for j in range(N):
            #print("Añadiendo termino: ", 1,("x",initial_node,j) ," a la primera restriccion")
            first_constrain.add_term(-1,("x",initial_node,j))
            #print(first_constrain)
        first_constrain.power(2)

        #print("La primera restriccion es: ", first_constrain)
        
        second_constrain = BinPol(var_shape_set)
        second_constrain.set_term(1,())
        for i in range(N):
            #print("Añadiendo termino: ", 1,("x",i,final_node) ," a la segunda restriccion")
            second_constrain.add_term(-1,("x",i,final_node))
            #print(second_constrain)
        second_constrain.power(2)

        #print("La segunda restriccion es: ", second_constrain)

        third_constrain = BinPol(var_shape_set)
        for k in range(N):
            if k != initial_node and k != final_node:
                third_constrain_aux = BinPol(var_shape_set)

                for i in range(N):
                    #print("añadiendo termino: ", 1,("x",i,k) ," a la tercera restriccion")
                    third_constrain_aux.add_term(1,("x",i,k))
                    #print(third_constrain_aux)
                for j in range(N):
                    #print("añadiendo termino: ", 1,("x",k,j) ," a la tercera restriccion")
                    third_constrain_aux.add_term(-1,("x",k,j))
                    #print(third_constrain_aux)
                third_constrain_aux.power(2)
                #print("La tercera restriccion auxiliar es: ", third_constrain_aux)
                third_constrain = third_constrain + third_constrain_aux
                #print("La tercera restriccion es: ", third_constrain)
                
    
        #print("La tercera restriccion es: ", third_constrain)
        ############################################################################################################
        fourth_constrain = BinPol(var_shape_set)
        termino_0 = BinPol(var_shape_set)
        for i in range(N):
                for j in range(N):
                    if j != final_node and j != i and j != initial_node:
                        fourth_constrain_aux = BinPol(var_shape_set)
                        #print("añadiendo termino: ", 1,("x",i,j) ," a la cuarta restriccion")
                        fourth_constrain_aux.add_term(1,("x",i,j))
                        #print(fourth_constrain_aux)
                        for k in range(N):
                            if k != i and fourth_constrain_aux != termino_0:
                                #print("añadiendo termino2: ", -1,("x",j,k) ," a la cuarta restriccion")
                                fourth_constrain_aux.add_term(-1,("x",j,k))
                                #print(fourth_constrain_aux)
                        fourth_constrain_aux.power(2)
                        #print("La cuarta restriccion auxiliar es: ", fourth_constrain_aux)
                        fourth_constrain = fourth_constrain + fourth_constrain_aux
                        #print("La cuarta restriccion es: ", fourth_constrain)

        #print("La cuarta restriccion es: ", fourth_constrain)
                    

        ############################################################################################################
        variable_constrain = BinPol(var_shape_set)
        for function in functions_list:      
            #print("function: ", function)  
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
            #print(nodes_with_functions)
            if function in initial_node_functions:
                variable_constrain_aux.add_term(1,()) 
            variable_constrain_aux.add_slack_variable('slack_variable_'+str(function), factor=-1)
            variable_constrain_aux.power(2)
            variable_constrain = variable_constrain + variable_constrain_aux
        
        #print("La cuarta restriccion es: ", variable_constrain)
        
        alpha1 = alpha1 * N
        alpha2 = alpha2 * N
        alpha3 = alpha3 * N
        alpha4 = alpha4 * N
        alpha5 = alpha5 * N

        fifth_constrain = BinPol(var_shape_set) 

        for a in range(N):
            fith_constrain_aux1 = BinPol(var_shape_set)
            fith_constrain_aux2 = BinPol(var_shape_set)
            for b in range(N):
                #print("añadiendo termino: ", 1,("x",a,b) ," a la quinta restriccion")
                fith_constrain_aux1.add_term(1,("x",a,b))
                #print("La auxiliar 1 es",fith_constrain_aux1)
            for c in range(N):
                #print("añadiendo termino: ", 1,("x",c,a) ," a la quinta restriccion")
                fith_constrain_aux2.add_term(1,("x",c,a))
                #print("La auxiliar 2 es",fith_constrain_aux2)
            fifth_constrain= fifth_constrain + (fith_constrain_aux1 * fith_constrain_aux2)
            #print("La quinta constrain es" ,fifth_constrain)    
        QUBOexpression =  cost_function + alpha1*first_constrain + alpha2*second_constrain + alpha3*third_constrain + alpha4*variable_constrain + alpha5*fifth_constrain
        
        return QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain , fifth_constrain, variable_constrain

            

                
                

            
        

    




