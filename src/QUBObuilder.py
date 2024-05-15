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
    def get_QUBO_model(self, graph, initial_node, final_node, functions, connections, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6):
        N = len(graph.nodes())
        A = 1
        problem_constant_bits = [[[-1 for _ in range(A)] for _ in range(N)] for _ in range(N)]

        for a in range(A):
            for i in range(N):
                for j in range(N):
                    if not graph.has_edge(i,j) or i == j:
                        problem_constant_bits[i][j][a] = 0
        
        #print(problem_constant_bits)
        problem_constant_bits_np = np.array(problem_constant_bits, dtype=np.int8)
        
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
        
        for i in range(N):
            lista_slack.append(VarSlack(name='slack_nodo_'+str(i),start=0,step=1,stop=2,slack_type=SlackType.binary))

        if A > 1:
            for i in range(N):
                for j in range(N):
                    if connections[i][j] != 0:
                        lista_slack.append(VarSlack(name='slack_bandwidth_'+str(i)+'_'+str(j),start=0,step=1,stop=connections[i][j]+1,slack_type=SlackType.binary))

        var_shape_set = VarShapeSet(BitArrayShape(name='x', shape=(N, N, A),axis_names=['i', 'j', 'a'], constant_bits=problem_constant_bits_np), *lista_slack)
        

        cost_function = BinPol(var_shape_set)
        first_constrain = BinPol(var_shape_set)
        second_constrain = BinPol(var_shape_set)
        third_constrain = BinPol(var_shape_set)
        fourth_constrain = BinPol(var_shape_set)
        variable_constrain = BinPol(var_shape_set)
        fifth_constrain = BinPol(var_shape_set) 
        sixth_constrain = BinPol(var_shape_set)
        alpha1 = 30
        alpha2 = 30
        alpha3 = 25 
        alpha4 = 40
        alpha5 = 40
        alpha6 = 2
        QUBOexpression = 0
        for a in range(A):
            
            for i in range(N):
                for j in range(N):
                    #print("Añadiendo termino: ", 1,("x",i,j) ," a la funcion de costo")
                    cost_function.add_term(1,("x",i,j,a))
                    #print(cost_function)

            #print("La funcion de costo es: ", cost_function)

            first_constrain_aux = BinPol(var_shape_set)
            first_constrain_aux.set_term(1,())
            for j in range(N):
                print("Añadiendo termino: ", 1,("x",initial_node,j,a) ," a la primera restriccion")
                first_constrain_aux.add_term(-1,("x",initial_node,j,a))
                print(first_constrain_aux)
            first_constrain_aux.power(2)
            first_constrain = first_constrain + first_constrain_aux
            print("La primera restriccion es: ", first_constrain)
            
            second_constrain_aux = BinPol(var_shape_set)
            second_constrain_aux.set_term(1,())
            for i in range(N):
                #print("Añadiendo termino: ", 1,("x",i,final_node) ," a la segunda restriccion")
                second_constrain_aux.add_term(-1,("x",i,final_node,a))
                #print(second_constrain_aux)
            second_constrain_aux.power(2)
            second_constrain = second_constrain + second_constrain_aux
            #print("La segunda restriccion es: ", second_constrain)

            
            for k in range(N):
                if k != initial_node and k != final_node:
                    third_constrain_aux = BinPol(var_shape_set)

                    for i in range(N):
                        #print("añadiendo termino: ", 1,("x",i,k) ," a la tercera restriccion")
                        third_constrain_aux.add_term(1,("x",i,k,a))
                        #print(third_constrain_aux)
                    for j in range(N):
                        #print("añadiendo termino: ", 1,("x",k,j) ," a la tercera restriccion")
                        third_constrain_aux.add_term(-1,("x",k,j,a))
                        #print(third_constrain_aux)
                    third_constrain_aux.power(2)
                    #print("La tercera restriccion auxiliar es: ", third_constrain_aux)
                    third_constrain = third_constrain + third_constrain_aux
                    #print("La tercera restriccion es: ", third_constrain)
                    
        
            #print("La tercera restriccion es: ", third_constrain)
            
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
                        variable_constrain_aux.add_term(1,("x",i,j,a))
                #print(nodes_with_functions)
                if function in initial_node_functions:
                    variable_constrain_aux.add_term(1,()) 
                variable_constrain_aux.add_slack_variable('slack_variable_'+str(function), factor=-1)
                variable_constrain_aux.power(2)
                variable_constrain = variable_constrain + variable_constrain_aux
            
            #print("La variable restriccion es: ", variable_constrain)

            #Constrain para asegurarse que no se vuelve a pasar por un nodo por el que ya hemos pasado
            for i in range(N):
                print("nodo: ", i)
                fourth_constrain_aux = BinPol(var_shape_set)
                fourth_constrain_aux.set_term(-1,())
                for j in range(N):
                    fourth_constrain_aux.add_term(1,("x",i,j,a))
                print("La cuarta restriccion auxiliar es: ", fourth_constrain_aux)
                fourth_constrain_aux.add_slack_variable('slack_nodo_'+str(i), factor=1)
                print("La cuarta restriccion auxiliar con slack es: ", fourth_constrain_aux)
                fourth_constrain_aux.power(2)
                print("La cuarta restriccion auxiliar con slack al cuadrado es: ", fourth_constrain_aux)
                fourth_constrain = fourth_constrain + fourth_constrain_aux
                print("La cuarta restriccion es: ", fourth_constrain)

            print("La cuarta restriccion es: ", fourth_constrain)
                        
            #Constrain para evitar ciclos de 2
            for i in range(N):
                for j in range(N):
                    fifth_constrain_aux = BinPol(var_shape_set)
                    fifth_constrain_aux2 = BinPol(var_shape_set)
                    print("Añadiendo termino: ", 1,("x",i,j,a) ," a la quinta restriccion")
                    fifth_constrain_aux.add_term(1,("x",i,j,a))
                    print("Añadiendo termino: ", 1,("x",j,i,a) ," a la quinta restriccion")
                    fifth_constrain_aux2.add_term(1,("x",j,i,a))

                    fifth_constrain_aux = fifth_constrain_aux * fifth_constrain_aux2
                    print("La quinta restriccion auxiliar es: ", fifth_constrain_aux)
                    fifth_constrain_aux.power(2)
                    print("La quinta restriccion auxiliar al cuadrado es: ", fifth_constrain_aux)

                    fifth_constrain = fifth_constrain + fifth_constrain_aux
                    print("La quinta restriccion es: ", fifth_constrain)
            
            #Constrain para el ancho de banda
            if A > 1:
                for i in range(N):
                    for j in range(N):
                        if connections[i][j] != 0:
                            sixth_constrain_aux = BinPol(var_shape_set)
                            sixth_constrain_aux.set_term(-1 * connections[i][j],())
                            sixth_constrain_aux.add_term(1,("x",i,j,a))
                            sixth_constrain_aux.add_slack_variable('slack_bandwidth_'+str(i)+'_'+str(j), factor=1)
                            sixth_constrain_aux.power(2)
                            sixth_constrain = sixth_constrain + sixth_constrain_aux
            
            QUBOexpression = QUBOexpression + cost_function + alpha1*first_constrain + alpha2*second_constrain + alpha3*third_constrain + alpha4*variable_constrain + alpha4*fourth_constrain + alpha5*fifth_constrain + 0*sixth_constrain
        
        return QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain, fourth_constrain,fifth_constrain, sixth_constrain, variable_constrain

            

                
                

            
        

    




