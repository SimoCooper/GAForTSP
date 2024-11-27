import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS
# 设置随机种子，确保每次运行结果可复现
np.random.seed(40)
random.seed(40)

# 生成城市之间距离矩阵
def generate_cities(num_cities):
    original_map = np.random.randint(low = 1, high = 20, size = (num_cities,num_cities)) * 2
    symmetric_map = (original_map + original_map.T)/2
    np.fill_diagonal(symmetric_map,np.inf)
    return symmetric_map

# 初始化种群，生成随机路径
def create_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        #随机打乱顺序，初始城市不固定
        random.shuffle(individual)
        population.append(individual)
    return np.array(population)


# 计算路径的适应度
def fitness_function(population, distance_matrix):
    fitness = []
    for individual in population:
        total_distance = 0
        #print(individual)
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i], individual[i + 1]]
        total_distance += distance_matrix[individual[-1], individual[0]]  # 最后还需要记录返回起点的路程
        fitness.append(1 / total_distance)  # 适应度为距离的倒数（此处可考虑使用不同的适应值函数）
    return np.array(fitness)

# 选择操作：基于适应度选择较好的个体进行交叉等操作.=
def selection(population, fitness, num_parents, index , distance_matrix):
    parents = []
    fitness_prob = fitness / fitness.sum()  # 将适应度标准化为概率分布
    if index == 'Roulette':
        for _ in range(num_parents):
            parent = population[np.random.choice(range(len(population)), p=fitness_prob)]
            parents.append(parent)
    elif index == 'Tournament':#这样会使得适应度高的个体被计算两次，可能陷入局部最优？
        Npop = len(population)
        # 创建一个随机排列
        pi = np.random.permutation(Npop)
        tparents = []
        for i in range(Npop - 1):
            idx1 = pi[i]
            idx2 = pi[i + 1]
            # 选择适应度较高的个体
            if fitness[idx1] > fitness[idx2]:
                tparents.append(population[idx1])
            else:
                tparents.append(population[idx2])
        # 最后一个锦标赛，将第一个和最后一个个体比较
        idx1 = pi[-1]
        idx2 = pi[0]
        if fitness[idx1] > fitness[idx2]:
            tparents.append(population[idx1])
        else:
            tparents.append(population[idx2])
        #print(parents)
        fitness = fitness_function(tparents, distance_matrix)
        sorted_indices = np.argsort(fitness)[::-1]  # 从大到小排序
        for i in range(num_parents):
            parents.append(tparents[sorted_indices[i]])
        # for _ in range(num_parents):
        #     selected_indices = np.random.choice(np.arange(len(population)), size=tournament_size, replace=False)
        #     # 选择适应度最高的个体
        #     best_index = selected_indices[np.argmax(fitness[selected_indices])]
        #     parents.append(population[best_index])
    elif index == 'Elitism':
        sorted_indices = np.argsort(fitness)[::-1]
        parents_indices = sorted_indices[:num_parents]
        parents = population[parents_indices]
    return np.array(parents)


# 交叉操作：部分匹配交叉
def crossover(parents, num_children):#采用部分匹配交叉，该帖子的演示十分直观：https://blog.csdn.net/hba646333407/article/details/103349279
    children = []
    # 对父系种群进行分割
    np.random.shuffle(parents)
    half_size = len(parents) // 2
    parent1 = parents[:half_size]
    parent2 = parents[half_size:]
    size = len(parents[0]) #其实就是城市的数量，只是懒得多传参数进去了
    for i in range((num_children // 2)):
        child1 = [-1] * size
        child2 = [-1] * size
        p1 = parent1[i]
        p2 = parent2[i]
        cxpoint1, cxpoint2 = sorted(np.random.choice(range(size), 2, replace=False))#选择两个位点
        child1[cxpoint1:cxpoint2+1] = p2[cxpoint1:cxpoint2+1]
        child2[cxpoint1:cxpoint2+1] = p1[cxpoint1:cxpoint2+1]
        #处理child1
        for j in range(cxpoint1, cxpoint2+1):
            val = child1[j]   #寻找父1中有没有相同的元素
            pos = np.where(p1 == val)[0][0]
            if (pos < cxpoint1 or pos > cxpoint2) and child1[pos] == -1:    #这说明这个元素在父1交叉部分以外
                #此时就要把它修改为对应的child2中的元素,但这仍然会可能发生重合
                while val in child1:#持续进行替换，直到换到child1中不会有重复的元素（一定能保证）
                    val = child2[np.where(child1 == val)[0][0]]
                child1[pos] = val #只有在这个元素在父1以外时才会需要进行更换，否则不需要
        # 处理child1
        for j in range(cxpoint1, cxpoint2+1):
            val = child2[j]
            pos = np.where(p2 == val)[0][0]
            if (pos < cxpoint1 or pos > cxpoint2) and child2[pos] == -1:
                while val in child2:
                    val = child1[np.where(child2 == val)[0][0]]
                child2[pos] = val
        # 填补剩余部分
        for j in range(size):
            if child1[j] == -1:
                child1[j] = p1[j]
            if child2[j] == -1:
                child2[j] = p2[j]
        print(child1,child2)
        children.extend([child1, child2])
    return np.array(children)


# 变异操作：交换随机两个城市的位置
def mutate(population, mutation_rate):
    for i in range(len(population)):
        if np.random.rand() < mutation_rate:
            swap_indices = np.random.choice(range(len(population[i])), size=2, replace=False)
            population[i][swap_indices[0]], population[i][swap_indices[1]] = population[i][swap_indices[1]], \
            population[i][swap_indices[0]]
    return population

# 计算路径总距离
def total_distance(path, distance_matrix):
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i], path[i + 1]]
    distance += distance_matrix[path[-1], path[0]]
    return distance

# 2-opt局部搜索,感觉这样做会耗费很多时间欸
@tf.function
def two_opt(path, distance_matrix):
    tf_path = tf.convert_to_tensor(path)
    tf_distance_matrix = tf.convert_to_tensor(distance_matrix)
    best_path = tf_path
    best_distance = tf.numpy_function(total_distance, [path, distance_matrix], tf.float64)

    for i in range(len(path) - 1):
        for j in range(i + 1, len(path)):
            # new_path = path.copy()
            # new_path[i:j + 1] = path[i:j + 1][::-1]
            # new_distance = total_distance(new_path, distance_matrix)
            new_path = tf.concat([tf_path[:i], tf.reverse(tf_path[i:j + 1], axis=[0]), tf_path[j + 1:]], axis=0)
            #new_distance = total_distance(new_path.numpy(), distance_matrix.numpy())
            new_distance = tf.numpy_function(total_distance, [new_path, distance_matrix], tf.float64)
            if new_distance < best_distance:
                best_path = new_path
                best_distance = new_distance
    return best_path

# 遗传算法主流程
def genetic_algorithm(num_cities ,num_generations, pop_size, num_parents, mutation_rate,
                      selection_index = 'Tournament'):
    #初始化城市以及路径
    distance_matrix = generate_cities(num_cities)
    population = create_population(pop_size, num_cities)
    best_distance = float('inf')
    best_solution = None
    best_generation = 0
    fitness = fitness_function(population, distance_matrix)
    for generation in range(num_generations):
        parents = selection(population, fitness, num_parents, selection_index, distance_matrix)
        children = crossover(parents, pop_size - num_parents)
        population = np.vstack([parents, children])
        #print(len(population))
        population = mutate(population, mutation_rate)

        # 应用2-opt局部搜索优化每个个体,太耗费时间了。。。
        # for i in range(len(population)):
        #     population[i] = two_opt(population[i], distance_matrix)
        fitness = fitness_function(population, distance_matrix)
        # 找到当前最优解
        current_best_fitness = fitness.max()
        current_best_idx = fitness.argmax()
        current_best_distance = 1 / current_best_fitness

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_solution = population[current_best_idx]
            best_generation = generation + 1
        print(f"Generation {generation + 1}:\n"
              f"Best Distance = {best_distance:.4f}\n"
              f"Best solution = {best_solution}")

    return best_solution, best_distance, best_generation, distance_matrix

# 可视化最佳路径
def plot_solution(distance_matrix, solution):
    np.fill_diagonal(distance_matrix, 0)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    # 根据最佳路径提取城市坐标
    path_coords = coords[solution]
    # 分离X和Y坐标
    x_coords, y_coords = path_coords[:, 0], path_coords[:, 1]
    # 绘制路径
    print(x_coords)
    print(y_coords)
    plt.figure(figsize=(16, 9))
    for i in range(len(x_coords) - 1):
        plt.arrow(x_coords[i], y_coords[i], x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i], head_width=0.6, head_length=0.6, fc='skyblue', ec='skyblue')
    # 绘制起点和终点
    plt.scatter(x_coords, y_coords, c='red')
    # 添加城市标签
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, f'{i}', fontsize=12, ha='right')
    # 添加路径标签
    plt.title('Best Path Visualization Based on Distance Matrix')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

# 运行遗传算法
best_solution, best_distance, best_generation, distance_matrix = genetic_algorithm(
    num_cities=10,  #200
    num_generations=100,  # 迭代次数
    pop_size=100,  # 种群大小500
    num_parents=60,  # 选择父代的数量,需保证其为偶数，且pop_size-num_parents也为偶数
    mutation_rate=0.02,  # 变异率
    selection_index = 'Tournament',#可供选择的有:
    #'Roulette'
    #'Tournament'
    #'Elitism'
)

# 输出结果
best_solution = np.concatenate([best_solution, [best_solution[0]]])  # 使得路径闭环
print("\nBest Solution = ", best_solution)
print("Best Distance = ", best_distance)
print("At Generation = ", best_generation)
plot_solution(distance_matrix, best_solution)
