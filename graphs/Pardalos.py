import re
import string
import numpy as np
import sys
import time
import operator


def isEdgeBetween(a, b):
    global graph
    return graph[a][b] == 1 if True else False


def pardalosPruneCondition(d, m, index):
    global max_clique
    return d + (m - index) <= len(max_clique)


def sortVerticesByDegrees():
    global graph
    degrees = {}
    for i in range(len(graph)):
        degrees[i] = graph[i].sum() - 1
    return [vertex[0] for vertex in sorted(degrees.items(), key=operator.itemgetter(1))]
    #return [vertex[1] for vertex in sorted(degrees.items(), reverse=True)]


def getSortedConnectedVertices(vertex, vertices, visited):
    global sortedVertices
    sortedConnectedVertices = []
    for v in [val for val in sortedVertices if val in vertices]:
        if v not in visited and isEdgeBetween(vertex, v):
            sortedConnectedVertices.append(v)
    return sortedConnectedVertices


def algorithm(vertices, depth, clique, visited):
    global max_clique
    for vertex in vertices:
        index = 1
        visited.append(vertex)
        new_visited = list(visited)
        new_clique = np.append(clique, vertex)
        if depth > len(max_clique):
            max_clique = list(new_clique)
            print(max_clique)
        if pardalosPruneCondition(depth, len(vertices), index):
            continue
        algorithm(getSortedConnectedVertices(vertex, vertices, new_visited), depth + 1, new_clique, new_visited)
        index += 1


if __name__ == "__main__":
    graph = []
    if len(sys.argv) > 2:
        timeout = int(sys.argv[2])

        file = sys.argv[1]
    else:
        file = "C:\\Users\\Daniil\\Desktop\\graphs\\MANN_a9.clq.txt"
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
    sortedVertices = sortVerticesByDegrees()
    algorithm(sortedVertices, 1, [], [])
    print(str(timeout - (end_time - time.time())) + " " + str(len(max_clique)))
