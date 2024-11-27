import numpy as np
import multiprocessing
from math import exp
from initialize import total_distance

T = 1500000
def two_opt(population, fitness,dis_matrix,uni_index,index,tabu):#uni_index表示的是进行two_option操作的个体在population的编号,不包含tabu
    position = -1
    size = len(population[0])
    max_two_opt = size // 5
    for k in uni_index:
        path = population[k]
        if index == 0:
            best_path = path
            best_dis = 1 / fitness[k]
            candidate_length  =  np.random.choice(range(size),np.random.randint(2, max_two_opt), replace=False)
            for length in candidate_length:
                for i in range(0, size - length):
                    new_path = path.copy()
                    new_path[i:i + length] = path[i:i + length][::-1]
                    new_dis = total_distance(new_path, dis_matrix)
                    if new_dis < best_dis:
                        best_path = new_path
                        best_dis = new_dis
            population[k] = best_path
            fitness[k] = 1 / best_dis
            tabu.append([path, best_path, 1 / best_dis])
        else:
            best_path = path
            best_dis = 1 / fitness[k]
            flag = 0
            length = np.random.randint(2, size)  # 有一个问题：这样一来每个个体的length长度是不一样的，这样似乎不大好？
            receive_path = []
            receive_dis = np.inf
            # 或许这样一来就不需要拿unique的个体来进行two_option了？即使是同样的个体，由于length的不同，其搜索的邻域也不同。
            # 因此对于数量比较多的个体（可以认为是适应度比较高的个体，因为适应度越高越容易被选择），通过多次在不同领域进行查找，可以让其在后续进化过程中获取“优势”
            for i in range(0, size - length):
                    new_path = path.copy()
                    new_path[i:i + length] = path[i:i + length][::-1]
                    new_dis = total_distance(new_path, dis_matrix)
                    if new_dis < best_dis:
                        best_path = new_path
                        best_dis = new_dis
                        flag += 1
                    # 在此处可引入类似于模拟退火算法中的接受度的概念，但若被接受，后续的best_path应仍为原path,若原path一直未找到更好的path再采用这个被接受的path
                    # 同时，若有两个distance比原有的path大的path被接受，那接下来它们也要根据模拟退火算法的准则再次进行比较，只保留一个
            #         else:
            #             receive_index = exp(
            #                 ((1 / new_dis) - (1 / best_dis)) * T)  # T取1，500，000时，从distance从5000变为5100的接受度为20%
            #             if np.random.random() < receive_index:
            #                 if new_dis < receive_dis:
            #                     receive_path = new_path
            #                     receive_dis = new_dis
            #                 else:
            #                     receive_index = exp(((1 / new_dis) - (1 / receive_dis)) * T)
            #                     if np.random.random() < receive_index:
            #                         receive_path = new_path
            #                         receive_dis = new_dis
            # if flag == 0:  # 即在搜索中没有找到最优个体
            #     if len(receive_path) != 0:  # 说明此时模拟退火找到了一个能接受的退化解
            #         population[k] = receive_path
            #         fitness[k] = 1 / receive_dis
            # else: #在搜索中找到了解
            population[k] = best_path
            fitness[k] = 1 / best_dis
    return population,fitness,tabu

def tabu_two_opt(population, fitness,dis_matrix,uni_index,index,tabu):#uni_index表示的是进行two_option操作的个体在population的编号
    position = -1
    size = len(population[0])
    for k in uni_index:
        path = population[k]
        for i, sublist in enumerate(tabu):  # 感觉在领域搜索的领域不确定时还是不使用禁忌搜索比较好
            if np.array_equiv(path, sublist[0]):
                index = 1
        # 当在禁忌表中找到个体时，不是返回找到的最优个体，而是会再次进行局部搜索（当某个个体第一次搜索时，保证找到最优个体；而后续找到次优个体）
        # 如不这样，易陷入局部最优（或许可以每次找到优秀个体的时候给它保存起来，然后当再一次搜索到它的时候从这些个体里随机选择一个进行返回）
        # 但这样或许会有过多的个体？
        if index == 0:
            best_path = path
            best_dis = 1 / fitness[k]
            for i in range(size - 1):
                for j in range(i + 2, size):
                    new_path = path.copy()  # 由于是在path的邻域找，所以不是best_path.copy(),防止best_path更新了之后跑去搜它的领域去了
                    new_path[i:j] = path[i:j][::-1]
                    new_distance = total_distance(new_path, dis_matrix)
                    if new_distance < best_dis:
                        best_path = new_path
                        best_dis = new_distance
            population[k] = best_path
            fitness[k] = 1 / best_dis
            tabu.append([path, best_path, 1 / best_dis])
        else:
            max_two_opt = size // 5
            best_path = path
            best_dis = 1 / fitness[k]
            flag = 0
            receive_path = []
            receive_dis = np.inf
            # 或许这样一来就不需要拿unique的个体来进行two_option了？即使是同样的个体，由于length的不同，其搜索的邻域也不同。
            # 因此对于数量比较多的个体（可以认为是适应度比较高的个体，因为适应度越高越容易被选择），通过多次在不同领域进行查找，可以让其在后续进化过程中获取“优势”
            candidate_length = np.random.choice(range(size), 2 * np.random.randint(1, max_two_opt),
                                                        replace=False)
            for length in candidate_length:
                for i in range(0, size - length):
                    new_path = path.copy()
                    new_path[i:i + length] = path[i:i + length][::-1]
                    new_dis = total_distance(new_path, dis_matrix)
                    if new_dis < best_dis:
                        best_path = new_path
                        best_dis = new_dis
                    # 在此处可引入类似于模拟退火算法中的接受度的概念，但若被接受，后续的best_path应仍为原path,若原path一直未找到更好的path再采用这个被接受的path
                    # 同时，若有两个distance比原有的path大的path被接受，那接下来它们也要根据模拟退火算法的准则再次进行比较，只保留一个
            #         else:
            #             receive_index = exp(
            #                 ((1 / new_dis) - (1 / best_dis)) * T)  # T取1，500，000时，从distance从5000变为5100的接受度为20%
            #             if np.random.random() < receive_index:
            #                 if new_dis < receive_dis:
            #                     receive_path = new_path
            #                     receive_dis = new_dis
            #                 else:
            #                     receive_index = exp(((1 / new_dis) - (1 / receive_dis)) * T)
            #                     if np.random.random() < receive_index:
            #                         receive_path = new_path
            #                         receive_dis = new_dis
            # if flag == 0:  # 即在搜索中没有找到最优个体
            #     if len(receive_path) != 0:  # 说明此时模拟退火找到了一个能接受的退化解
            #         population[k] = receive_path
            #         fitness[k] = 1 / receive_dis
            # else: #在搜索中找到了解
            population[k] = best_path
            fitness[k] = 1 / best_dis
    return population,fitness,tabu


#以下是专门为并行计算进行打造的
def parallel_two_opt(population, fitness,dis_matrix, uni_index,index,tabu,num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 创建任务列表，每个任务都是一个包含索引、种群和距离矩阵的元组
        #print('index=',index)
        tasks = [(i, population, fitness,dis_matrix,index,tabu) for i in uni_index]
        # 使用 pool.map 并行处理任务
        results = pool.map(process_path, tasks)
        # 更新种群
        for i, new_path,new_fitness,state in results:
            if state is True:
                tabu.append([population[i],new_path,new_fitness])
            population[i] = new_path
            fitness[i] = new_fitness
        return population,fitness,tabu

def process_path(args):
    i, population, fitness,d_matrix ,index, tabu= args
    best_path, best_fitness,state = basic_tabu_two_opt(population[i], fitness[i],d_matrix, tabu, index)
    return i, best_path, best_fitness,state

def basic_two_opt(path, fitness,dis_matrix,tabu,index = 0):#第三个返回值用来判断是否需要对禁忌表进行搜索
    position = -1
    size = len(path)
    max_two_opt = size // 5
#当在禁忌表中找到个体时，不是返回找到的最优个体，而是会再次进行局部搜索（第一次全局搜索找到最优个体，后续找到次优个体）
    #易陷入局部最优
    if index == 0:
        best_path = path
        best_dis = 1 / fitness
        candidate_length = np.random.choice(range(size), np.random.randint(2, max_two_opt), replace=False)
        for length in candidate_length:
            for i in range(0, size - length):
                new_path = path.copy()
                new_path[i:i + length] = path[i:i + length][::-1]
                new_dis = total_distance(new_path, dis_matrix)
                if new_dis < best_dis:
                    best_path = new_path
                    best_dis = new_dis
        return best_path, 1 / best_dis, False
    else:
        best_path = path
        best_dis = 1 / fitness
        flag = 0
        length = np.random.randint(2,len(path)) #有一个问题：这样一来每个个体的length长度是不一样的，这样似乎不大好？
        receive_path = []
        receive_dis = np.inf
        #或许这样一来就不需要拿unique的个体来进行two_option了？即使是同样的个体，由于length的不同，其搜索的邻域也不同。
        #因此对于数量比较多的个体（可以认为是适应度比较高的个体，因为适应度越高越容易被选择），通过多次在不同领域进行查找，可以让其在后续进化过程中获取“优势”
        for i in range(0,len(path) - length):
            new_path = path.copy()
            new_path[i:i + length] = path[i:i + length][::-1]
            new_dis = total_distance(new_path,dis_matrix)
            if new_dis < best_dis:
                best_path = new_path
                best_dis = new_dis
                flag += 1
        #在此处可引入类似于模拟退火算法中的接受度的概念，但若被接受，后续的best_path应仍为原path,若原path一直未找到更好的path再采用这个被接受的path
        #同时，若有两个distance比原有的path大的path被接受，那接下来它们也要根据模拟退火算法的准则再次进行比较，只保留一个
        #     else:
        #         receive_index = exp(((1 / new_dis) - (1 / best_dis) ) * T )#T取1，500，000时，从distance从5000变为5100的接受度为20%
        #         if np.random.random() < receive_index:
        #             if  new_dis < receive_dis:
        #                 receive_path = new_path
        #                 receive_dis = new_dis
        #             else:
        #                 receive_index = exp(((1 / new_dis) - (1 / receive_dis) ) * T )
        #                 if np.random.random() < receive_index:
        #                     receive_path = new_path
        #                     receive_dis = new_dis
        # if flag == 0 :#即在搜索中没有找到最优个体
        #     if len(receive_path) != 0:#说明此时模拟退火找到了一个能接受的退化解
        #         return receive_path, 1 / receive_dis, False
        #     else:
        #         return path, fitness, False

        return best_path, 1 / best_dis, False

def basic_tabu_two_opt(path, fitness,dis_matrix,tabu,index = 0):#第三个返回值用来判断是否需要对禁忌表进行搜索
    size = len(path)
    for i,sublist in enumerate(tabu):#感觉在领域搜索的领域不确定时还是不使用禁忌搜索比较好
        if np.array_equiv(path,sublist[0]):
            index = 1
            break
#当在禁忌表中找到个体时，不是返回找到的最优个体，而是会再次进行局部搜索（第一次全局搜索找到最优个体，后续找到次优个体）
    #易陷入局部最优
    if index == 0:
        best_path = path
        best_dis = 1 / fitness
        for i in range(size - 1):
            for j in range(i + 2, size):
                new_path = path.copy() #由于是在path的邻域找，所以不是best_path.copy(),防止best_path更新了之后跑去搜它的领域去了
                new_path[i:j] = path[i:j][::-1]
                new_distance = total_distance(new_path, dis_matrix)
                if new_distance < best_dis:
                    best_path = new_path
                    best_dis = new_distance
        return best_path, 1 / best_dis, True
    else:
        best_path = path
        best_dis = 1 / fitness
        max_two_opt = size // 10
        #或许这样一来就不需要拿unique的个体来进行two_option了？即使是同样的个体，由于length的不同，其搜索的邻域也不同。
        #因此对于数量比较多的个体（可以认为是适应度比较高的个体，因为适应度越高越容易被选择），通过多次在不同领域进行查找，可以让其在后续进化过程中获取“优势”
        candidate_length = np.random.choice(range(size), np.random.randint(2, max_two_opt), replace=False)
        for length in candidate_length:
            for i in range(0, size - length):
                new_path = path.copy()
                new_path[i:i + length] = path[i:i + length][::-1]
                new_dis = total_distance(new_path, dis_matrix)
                if new_dis < best_dis:
                    best_path = new_path
                    best_dis = new_dis
        #在此处可引入类似于模拟退火算法中的接受度的概念，但若被接受，后续的best_path应仍为原path,若原path一直未找到更好的path再采用这个被接受的path
        #同时，若有两个distance比原有的path大的path被接受，那接下来它们也要根据模拟退火算法的准则再次进行比较，只保留一个
        return best_path, 1 / best_dis, False



