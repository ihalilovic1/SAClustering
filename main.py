import math
import random
import numpy
import matplotlib.pyplot as plt

def distance(A, B):
    sqSum = 0
    for i in range(0, len(A)):
        sqSum += (A[i] - B[i])**2
    return sqSum**(0.5)

###
def AssignCluster(points, clusters):
    cp = []
    for p in points:
        min = 0
        for i in range(0, len(clusters)):
            if distance(p, clusters[i]) < distance(p, clusters[min]):
                min = i
        cp.append(min)
    return cp

### points[(x,y),...], initial_state[(x,y),...]
def SimulatedAnnealing(max_iter, initial_temperature, alpha, final_temperature, initial_state, points):
    t = initial_temperature
    current_state = initial_state[:]
    
    while(t >= final_temperature):
        for i in range(1, max_iter):
            next_state = perturb(current_state[:])
            energy_delta = value(next_state,points) - value(current_state,points)
            if ((energy_delta < 0) or (math.exp( -energy_delta / t) >= random.randint(0,10))):
                current_state = next_state
        t = alpha * t
    return current_state

def perturb(state):
    
    c = random.randint(0, len(state) - 1)
    d = 0.1
    G = numpy.random.normal()

    #for i in range(0, len(state[c])):
    state[c] = (state[c][0] + d*G, state[c][1] + d*G)   

    return state

def value(state, points):
    cp = AssignCluster(points, state)
    sum=0
    for i in range(0, len(points)):
        sum += distance(points[i], state[cp[i]])**2
    return sum


def test1():
    initial_state = [(2,6), (6,6), (9,3)]
    points = [(3,2), (3.5,2), (3.5, 2.5), (4,3), (7,4), (7.5,3.5), (8,5), (1.5,5)]

    final_state = SimulatedAnnealing(340, 50, 0.5, 0.05, initial_state, points)

    cp = AssignCluster(points, final_state)
    print(cp)

    plt.plot(initial_state[0][0], initial_state[0][1], 'bx')
    plt.plot(initial_state[1][0], initial_state[1][1], 'gx')
    plt.plot(initial_state[2][0], initial_state[2][1], 'rx')

    plt.plot(final_state[0][0], final_state[0][1], 'b*')
    plt.plot(final_state[1][0], final_state[1][1], 'g*')
    plt.plot(final_state[2][0], final_state[2][1], 'r*')
    for i in range(0, len(cp)):
        if cp[i] == 0:
            plt.plot(points[i][0], points[i][1], 'bo')
        elif cp[i] == 1:
            plt.plot(points[i][0], points[i][1], 'go')
        else:
            plt.plot(points[i][0], points[i][1], 'ro')
    plt.show()
test1()