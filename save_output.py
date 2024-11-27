import numpy as np
import matplotlib.pyplot as plt

def output_solution(file_name,best_solution, best_distance, best_generation,record_generation, record_distance,
                    record_runtime,record_accurate,record_unique):
    # 输出结果
    best_solution = np.concatenate([best_solution, [best_solution[0]]])  # 使得路径闭环
    # 画出图像
    plt.figure(figsize=(20, 20), dpi=100)

    plt.subplot(2, 2, 1)
    plt.plot(record_generation, record_distance, color='lime')
    for i in range(len(record_generation) // 10):
        plt.scatter(record_generation[10 * i], record_distance[10 * i], marker='.')
        plt.text(record_generation[10 * i],record_distance[10 * i] + 5,f"{int(record_distance[10 * i])}")
    plt.title('Best Distance Change through Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(record_generation, record_runtime, color='lime')
    for i in range(len(record_generation) // 10):
        plt.scatter(record_generation[10 * i], record_runtime[10 * i],marker='.')
        plt.text(record_generation[10 * i], record_runtime[10 * i] + 0.0008, f"{record_runtime[10 * i]:.2f}")
    plt.title('CPU Runtime through Generations')
    plt.xlabel('Generation')
    plt.ylabel('Runtime')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(record_generation, record_accurate, color='lime')
    for i in range(len(record_generation) // 10):
        plt.scatter(record_generation[10 * i], record_accurate[10 * i], marker='.')
        plt.text(record_generation[10 * i], record_accurate[10 * i] + 0.01, f"{(record_accurate[10 * i] * 100):.2f}%")
    plt.title('Precision through Generations')
    plt.xlabel('Generation')
    plt.ylabel('Precision')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(record_generation, record_unique, color='lime')
    for i in range(len(record_generation) // 10):
        plt.scatter(record_generation[10 * i], record_unique[10 * i], marker='.')
        plt.text(record_generation[10 * i], record_unique[10 * i] + 1, f"{int(record_unique[10 * i])}")
    plt.title('Unique through Generations')
    plt.xlabel('Generation')
    plt.ylabel('Unique')
    plt.grid(True)

    plt.savefig(f"results/{file_name[0]}_result.png")

    with open("results/results.txt", "a", encoding="utf_8") as f:
        f.write(120 * "#"+"\n"
                f"The results of problem {file_name[0]} :\n"
                f"Best Solution = {best_solution}\n"
                f"Best Distance = {best_distance}\n"
                f"Theoretical Best Distance = {file_name[1]:.4f}\n"
                f"Precision = {record_accurate[-1] * 100:.2f}%\n"
                f"At Generation = {best_generation}\n")
        f.close()


    #plt.show()
    print(f"The results of problem {file_name[0]} :")
    print("Best Solution = ", best_solution)
    print("Best Distance = ", best_distance)
    print(f"Theoretical Best Distance = {file_name[1]:.4f}")
    print(f"Precision = {record_accurate[-1] * 100:.2f}%")
    print("At Generation = ", best_generation)
    print(f"{file_name[0]} has done!!!\n\n\n")