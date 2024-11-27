import numpy as np
import pandas as pd
import random
import multiprocessing
from math import sqrt
from operator_crossover import pmx_multipoint
# 载入数据
def load_data(route):
    city_info = []
    with open(route,'r') as f:
        df = pd.read_csv(f, sep="\s+", skiprows=6, header=None)
        length = len(df) - 1
        city_name = np.array(df[0][0:length])  # 最后一行为EOF，不读入
        #city_name = city.tolist()
        city_x = np.array(df[1][0:length])
        city_y = np.array(df[2][0:length])
        for i in range(0,length) :
            city_info.append([city_name[i],city_x[i],city_y[i]])
    f.close()
    dis_map = np.zeros((length, length))
    for i in range(0, length):
        for j in range(0, length):  # 如果往返两城市代价一致可考虑从i+1开始遍历，然后map+=map.T
            dis_map[i][j] = sqrt((city_info[i][1] - city_info[j][1]) ** 2 + (city_info[i][2] - city_info[j][2]) ** 2)
    np.fill_diagonal(dis_map, np.inf)
    return length, dis_map

def calculate_distance(city):
    city_number = len(city)
    dis_map = np.zeros((city_number,city_number))
    for i in range(0,city_number):
        for j in range(0,city_number): #如果往返两城市代价一致可考虑从i+1开始遍历，然后map+=map.T
            dis_map[i][j] = sqrt( (city[i][1] - city[j][1])**2 + (city[i][2] - city[j][2]) ** 2)
    np.fill_diagonal(dis_map, np.inf)
    return city_number,dis_map

def greedy_path(begin_city, dis_matrix, num_cities):
    individual = [begin_city]
    now_city = begin_city
    for j in range(num_cities):
        dis_matrix[j][begin_city] = np.inf  # 保证不会选重复的节点
    for _ in range(1,num_cities):#第一个已经有了，因此不需要从0开始
        dis = np.inf
        city = now_city
        for i in range(num_cities):
            if dis_matrix[now_city][i] < dis:
                dis = dis_matrix[now_city][i]
                city = i
        for j in range(num_cities):
            dis_matrix[j][city] = np.inf #保证不会选重复的节点
        individual.append(city)
        now_city = city
    return individual

# 初始化种群，生成随机路径
def create_population(pop_size, num_cities,dis_matrix):
    population = []
    begin_city = list(range(num_cities))
    random.shuffle(begin_city)
    if num_cities > pop_size:
        for i in range(num_cities):
            dis_matrix_copy = dis_matrix.copy()
            population.append(greedy_path(begin_city[i],dis_matrix_copy, num_cities))
        fitness = fitness_function(population,dis_matrix)
        select_index = fitness.argsort()[::-1]
        select_index = select_index[:pop_size]
        population = np.array(population)
        population = population[select_index]
    else:
        for i in range(num_cities):
            dis_matrix_copy = dis_matrix.copy()
            population.append(greedy_path(begin_city[i],dis_matrix_copy, num_cities))
        population = np.array(population)
        cross_population = pmx_multipoint(np.array(population), pop_size - num_cities)
        population = np.vstack([population,cross_population])
        #下面是最原始的，完全随机打乱顺序的生成初始种群数量的方法
        # for i in range(num_cities,pop_size):
        #     individual = list(range(num_cities))
        #     #随机打乱顺序，初始城市不固定
        #     random.shuffle(individual)
        #     population.append(individual)
        #print(len(population))
        #下面是选取一部分个体用贪心算法进行生成
        # num_greedy = pop_size // 10  # 取1/10的个体通过贪心算法生成，注意此时不需要首尾相连
        # for i in range(num_greedy):
        #     dis_matrix_copy = dis_matrix.copy()
        #     population.append(greedy_path(begin_city[i],dis_matrix_copy, num_cities))
        # for i in range(num_greedy,pop_size):
        #     individual = list(range(num_cities))
        #     #随机打乱顺序，初始城市不固定
        #     random.shuffle(individual)
        #     population.append(individual)
    return population


# 计算路径的适应度
def fitness_function(population, dis_matrix):
    fitness = []
    for individual in population:
        total_dis = 0
        for i in range(len(individual) - 1):
            total_dis += dis_matrix[individual[i], individual[i + 1]]
        total_dis += dis_matrix[individual[-1], individual[0]]  # 最后还需要记录返回起点的路程
        fitness.append(1 / total_dis)  # 适应度为距离的倒数（此处可考虑使用不同的适应值函数）
    return np.array(fitness)

def fitness_windowing(self_fitness):
    self_fitness = np.array(self_fitness)
    if len(np.unique(self_fitness))  != 1:
        self_fitness = 1 / self_fitness
        beta_t = self_fitness.argmin()  # fitness differential,sigma scaling有点复杂了
        self_fitness -= self_fitness[beta_t] - 1
        self_fitness = 1 / self_fitness
    return self_fitness

def parallel_fitness_function(population, dis_matrix,num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 创建任务列表，每个任务都是一个包含索引、种群和距离矩阵的元组
        pop_length = len(population)
        fitness = np.zeros(pop_length)
        tasks = [(i, population, dis_matrix) for i in range(pop_length)]
        # 使用 pool.map 并行处理任务
        results = pool.map(process_fitness, tasks)
        # 更新种群
        for i, new_fitness in results:
            fitness[i] = new_fitness
        return np.array(fitness)

def process_fitness(args):
    i, population, d_matrix = args
    return i, 1 / total_distance(population[i], d_matrix)

def total_distance(path, dis_matrix):
    distance = 0
    for i in range(len(path) - 1):
        distance += dis_matrix[path[i], path[i + 1]]
    distance += dis_matrix[path[-1], path[0]]
    return distance

