#Author: Mario Guisado García
#Github: MarioGuisado

import numpy as np
from dadk.QUBOSolverCPU import *

def as_bqm(self) -> 'dimod.BinaryQuadraticModel':
    """
    The polynomial is returned as a :class:`dimod.BinaryQuadraticModel` object.

    :return: qubo as dimod.BinaryQuadraticModel object
    :rtype: dimod.BinaryQuadraticModel
    """

    try:
        import dimod
    except Exception as oops:
        print('\n\n' + (100 * '#'))
        print('pip install dwave-ocean-sdk')
        print((100 * '#') + '\n\n')
        raise oops

    return dimod.BinaryQuadraticModel(
        {i0: self._p1[i0] for i0 in self._p1},
        {(i0, i1): self._p2[i0][i1] for i0 in self._p2 for i1 in self._p2[i0]},
        self._p0,
        dimod.BINARY)

class QUBObuilder:
    def get_QUBO_model(self, graph, initial_node, final_node, functions, connections, resources,num_agents, alpha_cost_function):
        
        N = len(graph.nodes())
        A = num_agents
        problem_constant_bits = [[[-1 for _ in range(A)] for _ in range(N)] for _ in range(N)]
        
        for a in range(A):
            for i in range(N):
                for j in range(N):
                    if connections[i][j] == 0:
                        problem_constant_bits[i][j][a] = 0
        
        problem_constant_bits_np = np.array(problem_constant_bits, dtype=np.int8)
        
        function_values = functions.values()
        functions_list = [] 
        for value in function_values:
            if type(value) == tuple:
                for k in value:
                    if k not in functions_list:
                        functions_list.append(k)
            else:
                if value not in functions_list:
                    functions_list.append(value)
        
        initial_node_functions = []
        if initial_node in functions:
            for function in functions[initial_node]:
                initial_node_functions.append(function)

        slack_list = []
        for a in range(A):
            for i in functions_list:    
                if i in initial_node_functions:
                    slack_list.append(VarSlack(name='slack_variable_'+str(i)+"_"+str(a),start=0,step=1,stop=N-1,slack_type=SlackType.binary))
                else:
                    slack_list.append(VarSlack(name='slack_variable_'+str(i)+"_"+str(a),start=0,step=1,stop=N-2,slack_type=SlackType.binary))
            
        for a in range(A):
            for i in range(N):
                if i != initial_node and i != final_node:
                    slack_list.append(VarSlack(name='slack_nodo_'+str(i)+"_"+str(a),start=0,step=1,stop=2,slack_type=SlackType.binary))

        
        for i in range(N):
            for j in range(i+1,N):
                slack_list.append(VarSlack(name='slack_bandwidth_'+str(i)+'_'+str(j),start=0,step=1,stop=connections[i][j]+1,slack_type=SlackType.binary))

        for i in range(N):
            if i != initial_node and i != final_node:
                slack_list.append(VarSlack(name='slack_resources_node_'+str(i),start=0,step=1,stop=resources[i]+1,slack_type=SlackType.binary))

        var_shape_set = VarShapeSet(BitArrayShape(name='x', shape=(N, N, A),axis_names=['i', 'j', 'a'], constant_bits=problem_constant_bits_np), *slack_list)
        

        cost_function = BinPol(var_shape_set)
        first_constrain = BinPol(var_shape_set)
        second_constrain = BinPol(var_shape_set)
        third_constrain = BinPol(var_shape_set)
        fourth_constrain = BinPol(var_shape_set)
        variable_constrain = BinPol(var_shape_set)
        fifth_constrain = BinPol(var_shape_set) 
        sixth_constrain = BinPol(var_shape_set)
        seventh_constrain = BinPol(var_shape_set)

        for i in range(N):
            for j in range(i+1,N):
                    sixth_constrain_aux = BinPol(var_shape_set)
                    sixth_constrain_aux.set_term(-1 * (connections[i][j]),())
                    for a in range(A):                     
                        sixth_constrain_aux.add_term(1,("x",i,j,a))
                        sixth_constrain_aux.add_term(1,("x",j,i,a))
                    sixth_constrain_aux.add_slack_variable('slack_bandwidth_'+str(i)+'_'+str(j), factor=1)
                    sixth_constrain_aux.power(2)
                    sixth_constrain = sixth_constrain + sixth_constrain_aux
        
        for j in range(N):
            if j != initial_node and j != final_node:
                seventh_constrain_aux = BinPol(var_shape_set)
                seventh_constrain_aux.set_term(-1 * (resources[j]),())
                for i in range(N): 
                    for a in range(A):
                        seventh_constrain_aux.add_term(1,("x",i,j,a))
                seventh_constrain_aux.add_slack_variable('slack_resources_node_'+str(j), factor=1)
                seventh_constrain_aux.power(2)
                seventh_constrain = seventh_constrain + seventh_constrain_aux

        alpha1 = 30
        alpha2 = 30
        alpha3 = 30
        variable_alpha = 30
        alpha4 = 30
        alpha5 = 30
        alpha6 = 30
        alpha7 = 30
        QUBOexpression = 0

        for a in range(A):
            cost_function_aux = BinPol(var_shape_set)
            for i in range(N):
                for j in range(N):
                    cost_function_aux.add_term(1,("x",i,j,a))
            cost_function = cost_function + cost_function_aux

            first_constrain_aux = BinPol(var_shape_set)
            first_constrain_aux.set_term(1,())
            for j in range(N):
                first_constrain_aux.add_term(-1,("x",initial_node,j,a))
            first_constrain_aux.power(2)
            first_constrain = first_constrain + first_constrain_aux
            
            second_constrain_aux = BinPol(var_shape_set)
            second_constrain_aux.set_term(1,())
            for i in range(N):
                second_constrain_aux.add_term(-1,("x",i,final_node,a))
            second_constrain_aux.power(2)
            second_constrain = second_constrain + second_constrain_aux

            
            for k in range(N):
                if k != initial_node and k != final_node:
                    third_constrain_aux = BinPol(var_shape_set)

                    for i in range(N):
                        third_constrain_aux.add_term(1,("x",i,k,a))
                    for j in range(N):
                        third_constrain_aux.add_term(-1,("x",k,j,a))
                    third_constrain_aux.power(2)
                    third_constrain = third_constrain + third_constrain_aux
            
            for function_set in functions_list:      
                variable_constrain_aux = BinPol(var_shape_set)
                variable_constrain_aux.set_term(-1,()) 
                nodes_with_functions = []
                for node in functions:
                    function_value = functions[node]
                    if isinstance(function_value, tuple):
                        function_value = set(function_value)
                    elif not isinstance(function_value, set):
                        function_value = {function_value}
                    if function_set & function_value:
                        nodes_with_functions.append(node)
                for i in range(N):
                    for j in nodes_with_functions: 
                        variable_constrain_aux.add_term(1,("x",i,j,a))
                if function_set in initial_node_functions:
                    variable_constrain_aux.add_term(1,()) 
                variable_constrain_aux.add_slack_variable('slack_variable_'+str(function_set)+"_"+str(a), factor=-1)
                variable_constrain_aux.power(2)
                variable_constrain = variable_constrain + variable_constrain_aux
            
            for i in range(N):
                if i != initial_node and i != final_node:
                    fourth_constrain_aux = BinPol(var_shape_set)
                    fourth_constrain_aux.set_term(-1,())
                    for j in range(N):
                        fourth_constrain_aux.add_term(1,("x",i,j,a))
                    fourth_constrain_aux.add_slack_variable('slack_nodo_'+str(i)+"_"+str(a), factor=1)
                    fourth_constrain_aux.power(2)
                    fourth_constrain = fourth_constrain + fourth_constrain_aux

            for i in range(N):
                for j in range(N):
                    fifth_constrain_aux = BinPol(var_shape_set)
                    fifth_constrain_aux2 = BinPol(var_shape_set)
                    fifth_constrain_aux.add_term(1,("x",i,j,a))
                    fifth_constrain_aux2.add_term(1,("x",j,i,a))
                    fifth_constrain_aux = fifth_constrain_aux * fifth_constrain_aux2
                    fifth_constrain_aux.power(2)
                    fifth_constrain = fifth_constrain + fifth_constrain_aux
            
        
        QUBOexpression = alpha_cost_function*cost_function + alpha1*first_constrain + alpha2*second_constrain + alpha3*third_constrain + variable_alpha*variable_constrain + alpha4*fourth_constrain + alpha5*fifth_constrain + alpha6*sixth_constrain + alpha7*seventh_constrain
        return QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain, fourth_constrain,fifth_constrain, sixth_constrain, seventh_constrain, variable_constrain

    


                
                

            
        

    




