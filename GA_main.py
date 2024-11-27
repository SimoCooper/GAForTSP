###这是在原有GA代码的基础上进行的优化，主要有以下几个工作：1，引入标准的TSP问题的库，先解决200个城市左右的问题；
###2，对Memetic算法进行优化，引入禁忌搜索；
###3，引入并行计算，对算法进行加速
#下一步规划：1，在two_optional里引入模拟退火算法
#2，优化初始种群的配置，先用贪心算法生成一个初始种群

import numpy as np
import multiprocessing
import time
from operator_mutation import mutation
from operator_crossover import pmx,edge_crossover
from operator_memetic import two_opt,parallel_two_opt,tabu_two_opt
from operator_selection import selection
from save_output import output_solution
from initialize import load_data, create_population,fitness_function,parallel_fitness_function,fitness_windowing

def genetic_algorithm(train_data, num_generations, pop_size, crossover_rate, mutation_rate,
                               generation_gap,elite_rate, num_cpu, selection_index='Tournament', is_parallel= False):
    # 初始化城市以及路径
    num_cities, distance_matrix = load_data(f"data/{train_data[0]}")  # 其最优路径的距离为2579
    population = create_population(pop_size, num_cities, distance_matrix)
    best_ever_distance = float('inf')
    best_distance = float('inf')
    best_ever_solution = None
    best_solution = None
    best_generation = 0
    best_ever_generation = 0
    precision = 0
    if is_parallel is False:
        fitness = fitness_function(population, distance_matrix)
    else:
        fitness = parallel_fitness_function(population, distance_matrix,num_cpu)
    low_gradient_index = 0
    overall_tabu = []  # [0]储存原数组，[1]储存领域找到的最佳数组，[2]储存最佳适应度
    num_parents = int(pop_size * crossover_rate)
    num_replace = int(pop_size * generation_gap)
    num_elite = int(pop_size * elite_rate)

    record_current_distance = []
    record_generation = []
    record_runtime = []
    record_unique = []
    record_precision = []
    for generation in range(num_generations):
        initial_time = time.perf_counter()
        parents = selection(population, fitness_windowing(fitness), num_parents, selection_index, distance_matrix)
        #在进行选择操作时优化fitness
        children = pmx(parents, 7 * num_parents)  # 感觉产生后代的数量应该要为num_parents才对，而不是pop_size - num_parents
        #children = edge_crossover(parents,7 * num_parents,distance_matrix)
        # cross+mutation,cross+local（大概率该版本效果好）,三者都加，考虑顺序
        if is_parallel is False:
            children_fitness = fitness_function(children,distance_matrix)
        else:
            children_fitness = parallel_fitness_function(children,distance_matrix,num_cpu)
        # print('children:',len(children),'children_fitness:',len(children_fitness))
        # # 拿适应值大的个体再次进行局部搜索以替换适应值小的个体（有待考虑）
        # print(num_replace)
        unique_children, uni_index = np.unique(children, return_index=True, axis=0)
        temp_fitness = children_fitness[uni_index]
        max_index = temp_fitness.argsort()[::-1]
        replace_num = min(num_replace, len(uni_index))
        for i in range(replace_num):
            current_worst_idx = fitness.argmin()
            population[current_worst_idx] = unique_children[max_index[i]]
            fitness[current_worst_idx] = temp_fitness[max_index[i]]
        # 应用2-opt局部搜索优化每个个体
        _, uni_index = np.unique(population,return_index= True,axis= 0)
        if len(uni_index) > num_elite:
            uni_index = uni_index[:num_elite]
        # uni_index = current_best_idx[0:num_elite]
        #对于适应度高的个体，在其所有领域进行搜索,应该在领域搜索的时候就更新fitness，否则多次更新fitness太耗费时间了
        if is_parallel is False:
            population, fitness, overall_tabu = tabu_two_opt(population, fitness, distance_matrix, uni_index,
                                                             0, overall_tabu)  # index是告诉函数是否进行彻底的领域搜索，0表示彻底的搜索
        # 对于适应度高的个体，只在其部分领域内进行搜索（若问题规模小，可考虑全部使用所有邻域内搜索，但这样一来变异的用处似乎就小了）
        #uni_index = current_best_idx[num_elite:pop_size]
        else:
            population, fitness, overall_tabu = parallel_two_opt(population, fitness, distance_matrix, uni_index,
                                                             0, overall_tabu,num_workers=num_cpu)  # index是告诉函数是否进行彻底的领域搜索，0表示彻底的搜索
        uni_index = np.delete(range(pop_size),uni_index)
        if is_parallel is False:
            population, fitness, overall_tabu = tabu_two_opt(population, fitness, distance_matrix, uni_index,
                                                    1, overall_tabu)  # uni_index传入的是有哪些个体进行局部搜索
        else:
            population, fitness, overall_tabu = parallel_two_opt(population, fitness, distance_matrix, uni_index,
                                                                 1, overall_tabu, num_workers=num_cpu)
        current_best_fitness = fitness.max()
        current_best_idx = fitness.argmax()
        current_best_distance = 1 / current_best_fitness
        current_worst_idx = fitness.argmin()
        if current_best_distance < best_distance:
            if best_distance - current_best_distance < 5:
                low_gradient_index += 1
            else:
                low_gradient_index = 0
            best_distance = current_best_distance
            best_solution = population[current_best_idx]
            best_generation = generation + 1
            # if low_gradient_index >= 10:#当连续10代最优的距离变化较少时，应该要增大变异率以寻求跳出局部最优
            #     if mutation_rate < 0.1:
            #         mutation_rate += 0.01
            #         print('Increase mutation_rate for reason1')
            # else:
            #     if mutation_rate > 0.01:
            #         mutation_rate -= 0.01
            # print('Go back to normal mutation_rate')

        # 在过了若干代之后，如果最优距离仍然未变，可以有两种解决方式：1，保留已找到的最优个体，然后重启任务；2，增大变异率，当再次出现最优个体时变异率恢复原值
        # 前者相对来说更为激进,因此间隔的代数应该比较大，如50，100；后者相对而言比较温和，因此间隔的代数应该比较小，可选取5、10代
        # 而且上述的两者是可以并存的。（当diversity比较小的时候、连续n代改进比较小的时候也可以考虑增加变异率？）
        else:
            low_gradient_index = 0
            # if (generation - best_generation >= 100) and best_distance <= current_best_distance :
            #     population = create_population(pop_size, num_cities,distance_matrix)
            #     if best_ever_distance > best_distance:
            #         best_ever_distance = best_distance
            #         best_ever_solution = best_solution
            #         best_ever_generation = best_generation
            #     best_distance = float('inf')
            #     best_solution = None
            #     print("重启任务")
            if generation - best_generation >= 10:
                # if mutation_rate < 0.1:
                #     mutation_rate += 0.01
                #     print('Increase mutation_rate for reason2')
                if current_best_distance != best_distance:
                    population[current_worst_idx] = best_solution  # 防止 优良性状丢失，或许应该每次保留多一点？
        end_time = time.perf_counter()
        generation_time = end_time - initial_time
        precision = (best_distance - train_data[1]) / best_distance
        unique_population = np.unique(population, axis=0)

        record_generation.append(generation + 1)
        record_runtime.append(generation_time)
        record_current_distance.append(current_best_distance)
        record_unique.append(len(unique_population))
        record_precision.append(precision)
        print(f"Problem {train_data[0]}\n"
              f"Generation {generation + 1}:\n"
              # f"Best solution = {best_solution}\n"
              f"Theoretical Best Distance = {train_data[1]:.4f}\n"
              f"Best Distance = {best_distance:.4f}\n"
              f"Current Distance = {current_best_distance:.4f}\n"
              f"At generation = {best_generation}\n"
              # f"Tabu Array Length = {len(overall_tabu)}\n"
              # f"Mutation rate = {mutation_rate}\n"
              f"Precision = {precision:.4f}\n"
              f"The number of unique member = {len(unique_population)}\n",
              f"Run Time = {generation_time:.2f}s\n"
              ,flush = True)
        if ((generation - best_generation >= 20) and best_distance <= current_best_distance) or (precision <= 0.01):
            print("结束搜索")
            break
    if best_ever_solution is None:
        best_ever_distance = best_distance
        best_ever_solution = best_solution
        best_ever_generation = best_generation
    return (best_ever_solution, best_ever_distance, best_ever_generation, record_generation,
            record_current_distance, record_runtime, record_precision,record_unique)

#写智能控制函数，自动化该过程。能一次跑多个instance；500以下
# 画收敛曲线（横轴可以为CPU运行时间，两者都记录），画每一代的fitness及个体函数图像。landscape analysis(目标函数空间）
#改进顺序；gap改到3%以下。
#tsp推广到vrp（多算子选择，遍历性搜索）
#拓展到时间调度的优化scheduling
#进而拓展到更广的情况【多目标优化、多任务优化（迁移优化）（航线重合时可用）、动态优化】