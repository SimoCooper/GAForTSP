import numpy as np
# 变异操作：交换随机两个城市的位置
def mutation(population, mutation_rate):
    size = len(population[0])
    for i in range(len(population)):
        if np.random.rand() < mutation_rate:
            #下面是随机选若干变异点位的方法
            # max_mutation_point = size // 80
            # swap_indices = np.random.choice(range(len(population[i])), size=2 * max_mutation_point, replace=False)
            # random.shuffle(swap_indices)
            # for j in range(len(swap_indices)//2):
            #     temp =  population[i][swap_indices[2 * j]]
            #     population[i][swap_indices[2 * j]] =  population[i][swap_indices[2 * j + 1]]
            #     population[i][swap_indices[2 * j + 1]] = temp
            swap_indices = np.random.choice(range(len(population[i])), size=2 , replace=False)
            swap_indices = np.sort(swap_indices)
            #print(swap_indices)
            for j in range(0,(swap_indices[0] + swap_indices[1]) //2 - swap_indices[0] +1):
                temp = population[i][swap_indices[0] + j]
                population[i][swap_indices[0] + j] = population[i][swap_indices[1] - j]
                population[i][swap_indices[1] - j] = temp
    return population
