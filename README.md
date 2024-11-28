# GA For TSP
_Before browsing or downloading this project, please be aware that it is developed by an amateur and there are many aspects that need improvement_.

_Also, note that this README serves as a record of my development journey rather than a technical document. If you are looking for a technical document, please don't waste time on the following content_.  

___

## Quick Start

Once you have downloaded this project, place your data in the _data_ folder (your data should be in TSP format). Then, make a few revision in _self_run.py_. Your input data should be in the form of _[filename , theoretical best distance]_. If you have some knowledge of Python, these changes should be straightforward, so I won't go into too much detail here. Once you have made the necessary adjustments, you can find your result in the _results_ folder.

## Background

This project marks my initial venture into combinatorial optimization. Guided by my fellow apprentice, Guo, I chose to tackle the Travelling Salesman Problem(TSP) using Genetic Algorithm(GA), as both of them are classic and fundamental in the field. The data for this project is sourced from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/). Thanks to [_Introduction to Evolutionary Computing_](https://link.springer.com/book/10.1007/978-3-662-44874-8), I've learned about many useful operators and incorporated some of them in my  project. 

## Code Frameworks 

This project contains an initialization function, four operators, a GA function and a display function. If you want to modify any part of GA, simply locate and edit the corresponding function. Below is an introduction to each function in sequence.

## Initialization function

In fact, this function is a synthesis of many smaller functions that may be utilized after initialization. However, they are included here as they are essential for creating the first generation. Obviously, the _load_data_ function and _calculate_distance_ function are straightforward and do not require detailed explanation. In this section, I will discuss the _fitness_function_ and _create_population_ function.

### Fitness Function

As you can see, an individual's fitness is equal to the reciprocal of it's total distance. However, this can lead to an undesirable situation where, if the total distances of two individuals are both initially large, their fitness gap will be quite small, resulting in approximately equal selection rates. To address this, I used a windowing operation so that before selection, each fitness value is adjusted by subtracting _(the minimum fitness in population - 1)_. While _sigma scaling_ or _ranking selection_ could achieve similar effect, this method has proven to be sufficient for this project..

### Create Population Function

Initially, I create my population randomly. However, this approach led to slow evolution and poor precision. To address these drawbacks, I applied a greedy algorithm to the _create_population_ function. In this approach, each city is chosen as the departure city to generate a path. If the population size is smaller than the number of cities, we select the best of these paths to form the first generation. In cases where the population size is larger, we use the generated paths as parents to reproduce new paths until the number of paths reaches the desired population size. 

## Four Operators

Actually, there are three operators in a traditional GA. Since I've used a memetic algorithm in this project, I treat it as the fourth operators.  Let's take a look at these operators.

### Selection

There are numerous ways to perform selection, but they can broadly be classified into selection by fitness and selection by rank. In this project, all the selection operators are based on fitness rather than rank. This project includes five selection operators. The simplest one is roulette selection, while the tournament selection offers the best performance.

### Crossover

In TSP, it is essential to ensure that the individuals in the next generation do not have repeated cities. Therefore, designing suitable and sophisticated crossover operators is vital. I've developed two operators based on the book I mentioned earlier. It turns out that Partial Mapped Crossover(PMX) performs slightly better than Edge Crossover in my project. However, I believe Edge Crossover should be better because _it can maintain common edges between parents..._ I'm still trying to figure out why this isn't the case.

### Mutation & Memetic Algorithm

The reason I put them together is that they perform the same operations on an individual. They randomly select two point, then reverse these two points and all the points within. However, the memetic algorithm searches the entire neighborhood, whereas the mutation operator typically only search once per individual which means the former is more time consuming. If you find it inefficient to search the whole neighborhood, you can try narrowing down the search space. However, abandoning the memetic algorithm is unwise, as it could result in slight or even no improvement in the shortest distance.

## GA Function

In the _GA_main.py_, the GA is constructed. First, the data is loaded and used to calculate the distance matrix. Then, we generate our first generation. Once this is done, we enter a loop to continuously select parents and use them to reproduce the next generation. After that, we use some of the best offspring to replace the worst individuals. ~~And the new generation will go through mutation. (It turns out that its existence makes no contribution to the outcome)~~ At this point, we have implemented the traditional GA, but now the memetic algorithm comes into play. It's important to emphasize that the memetic algorithm consumes **a lot of time**. To address this, I've used multiprocessing to handle the task. If you don't want to use multiprocessing, or your computer doesn't support it, simply set *num_CPU* to *1* to run the algorithm in a single process. If the best distance doesn't change for *20* generations, the loop will be terminated.

## Display Function

This function will output the results to the _result_ folder and generate images to illustrate the changes in best distance, run time, precision and the number of unique individuals. The _lime_ color has been carefully selected and is truly beautiful.

***

## Postscript

Originally, I intended to write both a Chinese and an English version of the README. However, after writing extensively in English, I don't want to write a Chinese version anymore (because I'm lazy and I'm sure not so many people will click in this project...). I believe my explanations are straightforward and transparent. Additionally, all my code is annotated in Chinese, so there is no need to write this README in Chinese as well 

_Please note once again that this project is totally immature, and don't use it in any important project._

