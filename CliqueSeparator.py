import numpy as np
import networkx as nx
import networkx.algorithms.approximation.clique as nxclique
from math import log10
import time


class CliqueSeparator:
    def __init__(self, graph):
        self.graph = graph  # networkx graph
        self.graph_filled = graph.copy()
        self.n = graph.number_of_nodes()
        self.l = [1] * self.n  # associated vertex number
        self.alpha = [-1] * self.n  # ordering
        self.k = 1
        self.atoms = []
        self.separated = []  # used for coloring, contains arrays of AC, BC, C
        self.unnumbered = list(range(self.n))  # set of vertex names which hasn't number yet

    def lexM(self):
        for i in range(self.n - 1, -1, -1):
            reached = [True] * self.n
            reach = [[] for y in range(self.k + 1)]  # np.empty((self.k+1, ))
            u = -1
            for j in self.unnumbered:
                if self.l[j] == self.k and u == -1:
                    u = j
                else:
                    reached[j] = False
            if u == -1:
                return
            self.alpha[u] = i
            self.unnumbered.remove(u)
            self.l[u] = -1  # we don't need l-value of numbered vertex
            for w in [w for w in self.graph_filled.neighbors(u) if self.alpha[w] == -1]:
                reach[self.l[w]].append(w)
                reached[w] = True
                self.l[w] += 0.5
            self.search(reach, reached, u)
            self.sort()
        return sorted(range(self.n), key=lambda x: self.alpha[x])  # returns minimal ordering of vertexes

    def search(self, reach, reached, u):
        for j in range(1, self.k + 1):
            while len(reach[j]) != 0:
                w = reach[j].pop(0)
                for z in [z for z in self.graph_filled.neighbors(w) if reached[z] is False]:
                    reached[z] = True
                    if self.l[z] > j:
                        reach[self.l[z]].append(z)
                        self.l[z] += 0.5
                        self.graph_filled.add_edge(u, z)
                    else:
                        reach[j].append(z)

    def sort(self):
        k_values = sorted(set(l_val for l_val in self.l if l_val != -1))
        self.k = len(k_values)
        for index, value in enumerate(self.l, 0):
            if value != -1:
                self.l[index] = k_values.index(value) + 1
        self.unnumbered = sorted(self.unnumbered, key=lambda x: self.l[x])

    def separate(self, ordered, graph_filled):
        for u in ordered:
            c = [w for w in graph_filled.neighbors(u) if self.alpha[u] < self.alpha[w]]
            if self.is_clique_in_G(c):
                B_and_C = [v for v in ordered if v != u and (v in c or v not in graph_filled.neighbors(u))]
                if len(B_and_C) > len(c):
                    A = [v for v in ordered if v not in B_and_C]
                    atom = self.graph.copy()
                    atom.remove_nodes_from([v for v in range(self.n) if v not in A and v not in c])
                    self.atoms.append(atom)
                    # uncomment for coloring
                    # bc_graph = self.graph.copy()
                    # bc_graph.remove_nodes_from([v for v in range(self.n) if v not in B_and_C])
                    graph_filled_copy = graph_filled.copy()
                    graph_filled_copy.remove_nodes_from(A)

                    # self.separated.append([atom, bc_graph, c])
                    self.separate(B_and_C, graph_filled_copy)
                    return
        atom = self.graph.copy()
        atom.remove_nodes_from([v for v in range(self.n) if v not in graph_filled.nodes()])
        self.atoms.append(atom)

    def is_clique_in_G(self, vertexes):
        for i in range(len(vertexes)):
            for j in range(i + 1, len(vertexes)):
                if not self.graph.has_edge(vertexes[i], vertexes[j]):
                    return False
        return True

    def find_max_clique(self):
        current_max_clique_size = 1
        current_max_clique = []
        for atom in sorted(self.atoms, key=lambda x: len(x.nodes()), reverse=True):
            if len(atom.nodes()) <= current_max_clique_size:
                return current_max_clique
            clique = nxclique.max_clique(atom)
            if len(clique) > current_max_clique_size:
                current_max_clique = clique
                current_max_clique_size = len(clique)
        return current_max_clique

    def min_coloring(self):
        colors = None
        for values in reversed(self.separated):
            colors = self.merge_colors(values[0], values[1], values[2], colors)
        return colors

    def merge_colors(self, ac, bc, c, bc_colors):
        ac_colors = nx.coloring.greedy_color(ac, strategy=nx.coloring.strategy_largest_first)
        if bc_colors is None:
            bc_colors = nx.coloring.greedy_color(bc, strategy=nx.coloring.strategy_largest_first)

        c_in_a_colors = {vertex: ac_colors[vertex] for vertex in c}
        c_in_b_colors = {vertex: bc_colors[vertex] for vertex in c}
        if len(set(ac_colors.values())) < len(set(bc_colors.values())):
            ac_colors, bc_colors = bc_colors, ac_colors
        colors_to_be_replaced = {c_in_a_color: c_in_b_colors[key] for key, c_in_a_color in c_in_a_colors.items()}

        for vertex, color in bc_colors.items():
            if color in colors_to_be_replaced:
                bc_colors[vertex] = colors_to_be_replaced[color]
        bc_colors.update(ac_colors)
        return bc_colors


class Utils:
    @staticmethod
    def adj(graph, u):
        return np.where(graph[u] == 1)[0]

    @staticmethod
    def read_dimax(file_name):
        with open(file_name, 'r') as read_file:
            edges = []
            for line in read_file:
                if line[0] == 'e':
                    edge = tuple(int(x) - 1 for x in line[1:].split())
                    # graph[edge] = 1
                    edges.append(edge)
            G = nx.Graph()
            G.add_edges_from(edges)
        return G

    @staticmethod
    def get_digit(number, base, pos):
        return (number // base ** pos) % base

    @staticmethod
    def prefix_sum(array):
        for i in range(1, len(array)):
            array[i] = array[i] + array[i - 1]
        return array

    @staticmethod
    def radix_sort(l, base=10):
        passes = int(log10(max(l)) + 1)
        output = [0] * len(l)

        for pos in range(passes):
            count = [0] * base

            for i in l:
                digit = Utils.get_digit(i, base, pos)
                count[digit] += 1

            count = Utils.prefix_sum(count)

            for i in reversed(l):
                digit = Utils.get_digit(i, base, pos)
                count[digit] -= 1
                new_pos = count[digit]
                output[new_pos] = i

            l = list(output)
        return output


if __name__ == "__main__":
    G = Utils.read_dimax("./graphs/hamming10-2.clq.txt")
    clique_separator = CliqueSeparator(G)

    # counting decomposition time
    start_time = time.time()
    clique_separator.separate(clique_separator.lexM(), clique_separator.graph_filled)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(len(clique_separator.atoms) - 1)
