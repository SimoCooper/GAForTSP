import numpy as np

def pmx(parents, num_children):#采用部分匹配交叉，该帖子的演示十分直观：https://blog.csdn.net/hba646333407/article/details/103349279
    children = []
    num_remain = num_children
    half_size = len(parents) // 2
    size = len(parents[0])  # 其实就是城市的数量，只是懒得多传参数进去了
    #print("half_size = ",half_size)
    while num_remain > 0:
        # 对父系种群进行分割
        np.random.shuffle(parents)
        parent1 = parents[:half_size]
        parent2 = parents[half_size:]
        iterable_time = min(half_size,(num_remain // 2))
        for i in range(iterable_time):
            child1 = [-1] * size
            child2 = [-1] * size
            p1 = parent1[i]
            p2 = parent2[i]
            cross_point1, cross_point2 = sorted(np.random.choice(range(size), 2, replace=False))#其实可以生成2n个节点进行交叉互换，增加交叉效率
            child1[cross_point1:cross_point2+1] = p2[cross_point1:cross_point2+1]
            child2[cross_point1:cross_point2+1] = p1[cross_point1:cross_point2+1]
            #处理child1
            for j in range(cross_point1, cross_point2+1):
                val = child1[j]   #寻找父1中有没有相同的元素
                pos = np.where(p1 == val)[0][0]
                if (pos < cross_point1 or pos > cross_point2) and child1[pos] == -1:    #这说明这个元素在父1交叉部分以外,严格来说应该只需要后面那一个条件就够了
                    #此时就要把它修改为对应的child2中的元素,但这仍然会可能发生重合，因此要持续进行替换
                    while val in child1:#持续进行替换，直到换到child1中不会有重复的元素（一定能保证）
                        val = child2[np.where(child1 == val)[0][0]]
                    child1[pos] = val #只有在这个元素在父1以外时才会需要进行更换，否则不需要
            # 处理child2（其实可以与child1的处理合并在一起，加快运行速度）
            for j in range(cross_point1, cross_point2+1):
                val = child2[j]
                pos = np.where(p2 == val)[0][0]
                if (pos < cross_point1 or pos > cross_point2) and child2[pos] == -1:
                    while val in child2:
                        val = child1[np.where(child2 == val)[0][0]]
                    child2[pos] = val
            # 填补剩余部分
            for j in range(size):
                if child1[j] == -1:
                    child1[j] = p1[j]
                if child2[j] == -1:
                    child2[j] = p2[j]
            children.extend([child1, child2])
        #print(num_remain)
        num_remain -= half_size * 2
    return np.array(children)

def pmx_multipoint(parents, num_children):#考虑到亲代数量小于子代数量的情况，同时增多了交叉的点位
    children = []
    num_remain = num_children
    half_size = len(parents) // 2
    size = len(parents[0])  # 其实就是城市的数量，只是懒得多传参数进去了
    max_cross_point = size // 40
    while num_remain > 0:
        # 对父系种群进行分割
        np.random.shuffle(parents)
        parent1 = parents[:half_size]
        parent2 = parents[half_size:]
        iterable_time = min(half_size,(num_remain // 2))#对于PMX(Partially Mapped Crossover而言，两个父代一次交叉产生两个子代）
        for i in range(iterable_time):
            child1 = [-1] * size
            child2 = [-1] * size
            p1 = parent1[i]
            p2 = parent2[i]
            candidate_point = sorted(np.random.choice(range(size), 2 * np.random.randint(1,max_cross_point), replace=False))
            cross_point1 = []
            cross_point2 = []
            for j in range(len(candidate_point)//2):
                cross_point1.append(candidate_point[2 * j])
                cross_point2.append(candidate_point[2 * j + 1])
                child1[cross_point1[j]:cross_point2[j]+1] = p2[cross_point1[j]:cross_point2[j]+1]
                child2[cross_point1[j]:cross_point2[j]+1] = p1[cross_point1[j]:cross_point2[j]+1]
            for k in range(len(cross_point1)):
                #处理child1
                for j in range(cross_point1[k], cross_point2[k]+1):
                    val = child1[j]   #寻找父1中有没有相同的元素
                    pos = np.where(p1 == val)[0][0]
                    if child1[pos] == -1:    #这说明这个元素在父1交叉部分以外,严格来说应该只需要后面那一个条件就够了
                        #此时就要把它修改为对应的child2中的元素,但这仍然会可能发生重合，因此要持续进行替换
                        while val in child1:#持续进行替换，直到换到child1中不会有重复的元素（一定能保证）
                            val = child2[np.where(child1 == val)[0][0]]
                        child1[pos] = val #只有在这个元素在父1以外时才会需要进行更换，否则不需要
                # 处理child2（其实可以与child1的处理合并在一起，加快运行速度）
                for j in range(cross_point1[k], cross_point2[k]+1):
                    val = child2[j]
                    pos = np.where(p2 == val)[0][0]
                    if child2[pos] == -1:
                        while val in child2:
                            val = child1[np.where(child2 == val)[0][0]]
                        child2[pos] = val
            # 填补剩余部分
            for j in range(size):
                if child1[j] == -1:
                    child1[j] = p1[j]
                if child2[j] == -1:
                    child2[j] = p2[j]
            children.extend([child1, child2])
        num_remain -= half_size * 2
    return np.array(children)

def edge_crossover(parents, num_children,dis_matrix):
    children = []
    num_remain = num_children#该参数记录的是还需要生成多少个子代
    half_size = len(parents) // 2
    size = len(parents[0])
    while num_remain > 0:
        # 对父系种群进行分割
        np.random.shuffle(parents)
        parent1 = parents[:half_size]
        parent2 = parents[half_size:]
        iterable_time = min(half_size, num_remain)#由于这种方法只生成一个个体，因此num_remain无需除以2
        for i in range(iterable_time):
            child = [-1] * size
            p1 = parent1[i]
            p2 = parent2[i]
            clean_edg_table = [None] * size
            edge_table =  clean_edg_table.copy()
            #也可以放在一个表里，但这样的话会浪费时间比较
            clean_used_table = [0] * size
            used_table = clean_used_table.copy()
            for j in range(size) :#根据p1和p2来生成edge_table
                if edge_table[p1[j]] is None:
                    edge_table[p1[j]] = [p1[j - 1], p1[(j + 1)%size]]
                else:
                    edge_table[p1[j]] = np.hstack([edge_table[p1[j]],[p1[j - 1], p1[(j + 1)%size]]])
                    uni_element, uni_index = np.unique(edge_table[p1[j]],return_counts=True)
                    uni_element = uni_element.astype('str')
                    for k in range(len(uni_index)):
                        if uni_index[k] != 1:
                            uni_element[k] = f"{uni_element[k]}+"
                    edge_table[p1[j]] = uni_element
                #下面对于p2[j]就是一样的操作了
                if edge_table[p2[j]] is None:
                    edge_table[p2[j]] = [p2[j - 1], p2[(j + 1)%size]]
                else:
                    edge_table[p2[j]] = np.hstack([edge_table[p2[j]],[p2[j - 1], p2[(j + 1)%size]]])
                    uni_element, uni_index = np.unique(edge_table[p2[j]],return_counts=True)
                    uni_element = uni_element.astype('str')
                    for k in range(len(uni_index)):
                        if uni_index[k] != 1:
                            uni_element[k] = f"{uni_element[k]}+"
                    edge_table[p2[j]] = uni_element

            #接下来构建子代的路径，注意edge_table存的都是字符串形式的，因此要转化
            current_element = int(np.random.choice(range(size), 1))
            child[0] = current_element
            used_table[current_element] += 1
            for j in range(size - 1):#因为已经确定了一个点了
                candidate = []
                edge = edge_table[current_element]
                edge = np.array(edge).astype('str')#现在所保存的元素的邻边
                #print(edge)
                if len(edge) == 3:
                    flag = 0
                    for k in edge:#current_element就是现在的点，在edge_table中要找其邻边
                        if "+" in k:#若该条件成立，则说明k是这两个表中共有的元素，即边
                            candidate_point = int(k[:len(k)-1])#除去最右边的"+"
                            if used_table[candidate_point] == 0:
                                current_element = candidate_point
                                flag = 1
                                break
                        else:#有可能公共边对应的节点被占了，因此还是得记录其他边
                            candidate_point = int(k)
                            if used_table[candidate_point] == 0:
                                candidate.append(candidate_point)
                    if flag == 0:
                        current_element = edge_choose(current_element,candidate,dis_matrix,used_table)
                elif len(edge) ==2:
                    for k in edge:#current_element就是现在的点，在edge_table中要找其邻边
                        candidate_point = int(k[:len(k)-1])#除去最右边的"+"
                        if used_table[candidate_point] == 0:
                            candidate.append(candidate_point)
                    current_element = edge_choose(current_element, candidate, dis_matrix, used_table)
                else:#只可能有2、3、4这三种情况
                    for k in edge:
                        candidate_point = int(k)
                        if used_table[candidate_point] == 0:
                            candidate.append(candidate_point)
                    current_element = edge_choose(current_element, candidate, dis_matrix, used_table)
                child[j + 1] = current_element
                used_table[current_element] += 1
            children.append(child)
        num_remain -= half_size
    return np.array(children)

def edge_choose(current_element,candidate, dis_matrix, used_table):
    if len(candidate) ==0:
        best_distance = np.inf
        best_element = current_element
        for i in range(len(used_table)):
            if used_table[i] ==0:
                current_dis = dis_matrix[current_element][i]
                if best_distance > current_dis:
                    best_element = i
                    best_distance = current_dis
        return int(best_element)
        #return int(np.random.choice(np.where(np.array(used_table) == 0)[0],1))
    else:
        best_dis = np.inf
        best_element =  current_element
        for i in candidate:#candidate里面保存的都是数字，可放心使用
            if used_table[i] ==0:
                current_dis = dis_matrix[current_element][i]
                if best_dis > current_dis:
                    best_element = i
                    best_dis = current_dis
                return int(best_element)

