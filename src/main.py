import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from initializer import * 
from QUBObuilder import *
from dwave.system import LeapHybridSampler
from constants import token
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="dadk.BinPol")


graph = nx.DiGraph()
initializer = Initializer(graph)

(connections, function_file_name, resources_file_name, hybrid_time, dot_size, alpha_cost_function) = initializer.get_file_data()
available_connections, connections = initializer.process_connections(connections)

for i, row in enumerate(connections):
    for j, value in enumerate(row):
        if value != 0:
            graph.add_edge(i, j)

initializer.draw()
builder = QUBObuilder()

functions = {}
desired_functions = {}

input_str = input("Please enter the desired functions, separated by spaces: ")
input_list = input_str.split(' ')
desired_functions = {int(num_str) for num_str in input_list}

with open(function_file_name, 'r') as f:
    nodo = 0
    for line in f:
        functions_aux = list(map(int, line.split()))
        for i in range(len(functions_aux)):
            if functions_aux[i] in desired_functions:
                if nodo not in functions:
                    functions[nodo] = set()
                functions[nodo].add(functions_aux[i])
        nodo += 1
       
with open(resources_file_name, 'r') as file:
    resources = file.readline().split()

resources = [int(resource) for resource in resources]

num_agents = initializer.get_num_agents()
start_node, end_node = initializer.get_start_end_nodes()

QUBOexpression, cost_function, first_constrain ,second_constrain ,third_constrain ,fourth_constrain,fifth_constrain, sixth_constrain,seventh_constrain, variable_constrain = builder.get_QUBO_model(graph, start_node, end_node, functions, connections,resources,num_agents,alpha_cost_function)
solver = QUBOSolverCPU(
number_iterations=200000,
number_runs=30,
scaling_bit_precision=16,
auto_tuning=AutoTuning.AUTO_SCALING_AND_SAMPLING)

user_input = input("Do you want to execute the simulator first (only recommended for small graphs and 1 or 2 agents, it may be inaccurate)?  (1 - yes / 0 - no): ")
if user_input.lower() == 1:
    solution_list = solver.minimize(QUBOexpression)
    configuration = solution_list.min_solution.configuration

    for p in cost_function, first_constrain, second_constrain ,third_constrain , fourth_constrain,fifth_constrain, sixth_constrain,seventh_constrain, variable_constrain :
        print("Min %s: at %s value %f" % (p, configuration, p.compute(configuration)))

    solution_list = solver.minimize(QUBOexpression)
    my_bit_array = solution_list.min_solution.extract_bit_array('x')
    my_bit_array.draw(axis_names=['i', 'j', 'a'])

        
bqm_problem=QUBOexpression.as_bqm()
os.environ['DWAVE_API_TOKEN']= token
sampler = LeapHybridSampler()

print("Executing Hybrid Solver...")
answer = sampler.sample(bqm_problem, time_limit=hybrid_time, label='Hybrid Exec')

samples = []
for datum in answer.data(['sample', 'energy']):
   first_N_bits = {k: v for k, v in datum.sample.items() if k < available_connections * num_agents}
   samples.append(first_N_bits)
   break

n = len(samples) * num_agents  
cols = int(np.ceil(np.sqrt(n)))
rows = int(np.ceil(n / cols))
fig = plt.figure(figsize=(cols*6, rows*6))

for i, sample in enumerate(samples):
    for agent in range(num_agents):
        matrix = np.zeros_like(connections)
        contador = 0
        for j, value in np.ndenumerate(connections):
            if value == 0:
                matrix[j] = 0
            else:
                matrix[j] = sample[contador*num_agents+agent]
                contador += 1

        ax = fig.add_subplot(rows, cols, i*num_agents+agent+1)

        for (x, y), value in np.ndenumerate(matrix):
            if connections[x][y] == 0:
                color = 'lightblue'
            elif value == 0:
                color = 'blue' 
            else:
                color = 'red'
            ax.scatter(x, y, c=color, s=dot_size) 

        ax.set_xticks(np.arange(matrix.shape[1]), np.arange(1, matrix.shape[1] + 1))
        ax.set_yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1))
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        ax.set_title(f'Sample {i+1}, Agent {agent+1}')

plt.show()