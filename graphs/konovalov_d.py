import re
import string
import numpy as np
import sys
import time


def canBeAddedToClique(graph, clique, v):
    for i in clique:
        if graph[i][v] == 0:
            return False
    return True


def countDegrees(graph, degrees):
    for i in range(len(graph)):
        degrees[i] = graph[i].sum()-1
        print(graph[i])
    print(degrees)
    return


def tryToIncreaseClique(graph, clique):
    global end_time, max_clique, degrees
    #if time.time() > end_time:
    #    sys.exit(str(0) + " " + str(max_clique))

    for i in range(np.amax(clique) + 1, len(graph)):
        if degrees[i] < len(max_clique):
            continue
        if canBeAddedToClique(graph, clique, i):
            new_clique = np.append(clique, i)
            tryToIncreaseClique(graph, new_clique)
    if (len(max_clique) < len(clique)):
        max_clique = list(clique)
    return


if __name__ == "__main__":
    graph = []
    if len(sys.argv) > 2:
        timeout = int(sys.argv[2])

        file = sys.argv[1]
    else:
        file="C:\\Users\\Daniil\\Desktop\\graphs\\MANN_a9.clq.txt"
        timeout = 3000
    end_time = time.time() + timeout
    with open(file, 'r') as read_file:
        for line in read_file:
            if line.startswith('p'):
                graph = np.eye(int(re.search('[0-9]+', line).group()) + 1)
            elif line.startswith('e'):
                e = re.findall('[0-9]+', line)
                graph[int(e[0])][int(e[1])] = 1
                graph[int(e[1])][int(e[0])] = 1

    max_clique = []
    degrees = np.zeros(len(graph))
    countDegrees(graph, degrees)
    for i in range(len(graph)):
        clique = [i]
        tryToIncreaseClique(graph, clique)
    print(str(timeout - (end_time - time.time())) + " " + str(len(max_clique)))
