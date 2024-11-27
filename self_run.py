from GA_main import genetic_algorithm
import multiprocessing
from save_output import output_solution

if __name__ == "__main__":
    multiprocessing.freeze_support()
    data = [["ch130.tsp",6110],["a280.tsp",2579],["lin318.tsp",42029],["fl417.tsp",11861],["u574.tsp",36905]]
    #data = [["ch130.tsp",6110]]
    #["gr202.tsp",40160]
    # 运行遗传算法
    #parallel_run(data,10)
    for i in range(5):
        best_sol, best_dist, best_gen, record_gen, record_dis, record_time, record_accurate, record_unique = genetic_algorithm(
            train_data=data[i],
            num_generations=1000,  # 迭代次数
            pop_size=100,  # 种群大小500 100。最好100
            crossover_rate=0.2,  # 选择父代的数量,需保证其为偶数，且pop_size-num_parents也为偶数，应该写rate,0.8,0.2,0.6,0.4.
            mutation_rate=0.01,  # 变异率
            generation_gap=0.05,  # 每次替换个体数占种群总个体数的百分比
            elite_rate=0.4,
            num_cpu=10,  # 被分配的CPU资源数
            selection_index='Tournament',  # 可供选择的有:
            # 'Roulette'
            # 'SUS'
            # 'Tournament'
            # 'Elitism'
            # 'Unbiased_Tournament'
            is_parallel=True,
        )
        output_solution(data[i], best_sol, best_dist, best_gen, record_gen, record_dis, record_time,
                        record_accurate, record_unique)