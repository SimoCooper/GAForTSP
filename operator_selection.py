import numpy as np
from initialize import fitness_function
# 选择操作：基于适应度选择较好的个体进行交叉等操作.=
def selection(population, self_fitness, num_selection, index , distance_matrix):
    parents = []
    fitness_prob = self_fitness / self_fitness.sum()  # 将适应度标准化为概率分布
    if index == 'Roulette':
        for _ in range(num_selection):
            parent = population[np.random.choice(range(len(population)), p=fitness_prob)]
            parents.append(parent)
    elif index == 'SUS':#stochastic universal sampling
        unique_population, position, counts = np.unique(population,return_index=True,return_counts=True,axis=0)
        temp_fitness = fitness_prob[position]
        expectation = temp_fitness * counts * num_selection
        num_remain = num_selection
        for i in range(len(expectation)):
            while (expectation[i] >= 1) & (num_remain >0):
                parents.append(unique_population[i])
                num_remain -= 1
                expectation[i] -= 1
        if expectation.sum !=0:
            fitness_prob = expectation / expectation.sum()
        for _ in range(num_remain):
            parent = unique_population[np.random.choice(range(len(unique_population)), p=fitness_prob)]
            parents.append(parent)
    elif index == 'Tournament':
        #上面的是unbiased_tournament,下面的是随机挑选的.前者会使得适应度高的个体被计算两次，可能陷入局部最优？
        #经过验证，发现用上面的方法，每次迭代的进步很小，在经过3000代迭代后，distance从30000降到了7000，且陷入了局部最优，再迭代3000代也只降到了6582
        #而用下面的方法，只需200代即可到7000
        for _ in range(num_selection):
            selected_indices = np.random.choice(np.arange(len(population)), size = 2 , replace = False)
            # 选择适应度最高的个体
            best_index = selected_indices[np.argmax(self_fitness[selected_indices])]
            parents.append(population[best_index])

    elif index == 'Elitism':
        sorted_indices = np.argsort(self_fitness)[::-1]
        parents_indices = sorted_indices[:num_selection]
        parents = population[parents_indices]

    elif index == 'Unbiased_Tournament':
        num_population = len(population)
        # 创建一个随机排列
        temp_parents = []
        num_remain = num_selection
        while num_remain > 0:
            pi = np.random.permutation(num_population)
            for i in range(num_population // 2):#应该在一次循环时每个元素只比较一次
                idx1 = pi[2 * i]
                idx2 = pi[2 * i + 1]
                # 选择适应度较高的个体
                if self_fitness[idx1] > self_fitness[idx2]:
                    temp_parents.append(population[idx1])
                else:
                    temp_parents.append(population[idx2])
            # 最后一个锦标赛，将第一个和最后一个个体比较
            idx1 = pi[-1]
            idx2 = pi[0]
            if self_fitness[idx1] > self_fitness[idx2]:
                temp_parents.append(population[idx1])
            else:
                temp_parents.append(population[idx2])
            num_remain -= num_population // 2
        self_fitness = fitness_function(temp_parents, distance_matrix)
        sorted_indices = np.argsort(self_fitness)[::-1]  # 从大到小排序
        for i in range(num_selection):
            parents.append(temp_parents[sorted_indices[i]])
    return np.array(parents)
