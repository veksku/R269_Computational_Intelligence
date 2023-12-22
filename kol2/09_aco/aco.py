import networkx as nx
import random
import math
from matplotlib import pyplot as plt
from tqdm import tqdm

def traverse(g, source_node, alpha, beta):
    visited = [False for _ in g.nodes]
    visited[source_node] = True

    cycle = [source_node]
    cycle_length = 0

    u = source_node
    for _ in range(len(g.nodes) - 1):
        # graf je kompletan
        neighbors = []
        neighbor_values = []
        for v in g.nodes:
            if not visited[v]:
                pheromone = g[u][v]['pheromone']
                distance = g[u][v]['distance']
                value = pheromone**alpha / distance**beta
                neighbors.append(v)
                neighbor_values.append(value)

        next_node = random.choices(population=neighbors, weights=neighbor_values, k=1)[0]
        visited[next_node] = True
        cycle.append(next_node)
        cycle_length += g[u][next_node]['distance']
        u = next_node

    # tezina grane od poslednjeg do prvog cvora u ciklusu
    cycle_length += g[cycle[-1]][cycle[0]]['distance']
    return cycle, cycle_length

def aco(g, num_iters, num_ants, alpha, beta, q, rho):
    best_cycle = None
    best_length = float('inf')

    for it in tqdm(range(num_iters)):
        cycles = [traverse(g=g, source_node=random.randrange(len(g.nodes)), alpha=alpha, beta=beta) for _ in range(num_ants)]
        # print(best_cycle)

        # isparavanje
        for u in range(len(g.nodes)):
            for v in range(u+1, len(g.nodes)):
                g[u][v]['pheromone'] *= rho

        # dodavanje feromona
        for cycle, cycle_length in cycles:
            if cycle_length < best_length:
                best_length = cycle_length
                best_cycle = cycle.copy()

            delta = q / cycle_length
            for u, v in zip(cycle[:-1], cycle[1:]):
                g[u][v]['pheromone'] += delta
            g[cycle[0]][cycle[-1]]['pheromone'] += delta

        if it == 0:
            drawing = draw_cycle(best_cycle, g, f'res/{it}.png')
        else:
            update_drawn_cycle(best_cycle, g, drawing)

    print(best_cycle)
    print(best_length)

def euclidean_distance(a_x, a_y, b_x, b_y):
    return math.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)

def read_graph(file_path):
    with open(file_path) as f:
        g = nx.Graph()
        for line in f:
            if not line[0].isdigit():
                continue
            node, x, y = line.split()
            node = int(node) - 1
            x = float(x)
            y = float(y)
            g.add_node(node, x=x, y=y)
        
        for u in g.nodes:
            for v in range(u + 1, len(g.nodes)):
                g.add_edge(u, v,
                           distance=euclidean_distance(g.nodes[u]['x'],
                                                        g.nodes[u]['y'],
                                                        g.nodes[v]['x'],
                                                        g.nodes[v]['y']),
                            pheromone=random.uniform(1e-5, 1e-2))
        
        return g

def get_xy_from_cycle(cycle, g):
    xs = []
    ys = []
    for node in cycle:
        xs.append(g.nodes[node]['x'])
        ys.append(g.nodes[node]['y'])
    xs.append(xs[0])
    ys.append(ys[0])
    return xs, ys

def draw_cycle(cycle, g, img_file_path):
    xs, ys = get_xy_from_cycle(cycle, g)
    drawing = plt.plot(xs, ys, marker='o')[0]
    # plt.savefig(img_file_path)
    plt.ion()
    plt.show()
    return drawing

def update_drawn_cycle(cycle, g, drawing):
    xs, ys = get_xy_from_cycle(cycle, g)
    drawing.set_data(xs, ys)
    plt.pause(0.05)

def main():
    g = read_graph(file_path='wi29.tsp')
    aco(g=g, num_iters=100, num_ants=10, alpha=0.9, beta=1.5, q=10, rho=0.9)
    
if __name__ == '__main__':
    main()