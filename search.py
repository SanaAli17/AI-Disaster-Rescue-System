from collections import deque
import heapq

def bfs(env, start, goal):
    queue = deque()
    queue.append((start, [start]))
    visited = set()

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path

        if (x, y) in visited:
            continue

        visited.add((x, y))

        for nx, ny in env.get_neighbors(x, y):
            queue.append(((nx, ny), path + [(nx, ny)]))

    return None

def dfs(env, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()

        if (x, y) == goal:
            return path

        if (x, y) in visited:
            continue

        visited.add((x, y))

        for nx, ny in env.get_neighbors(x, y):
            stack.append(((nx, ny), path + [(nx, ny)]))

    return None

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(env, start, goal):
    pq = []
    heapq.heappush(pq, (0, start, [start]))

    visited = set()

    while pq:
        cost, (x, y), path = heapq.heappop(pq)

        if (x, y) == goal:
            return path

        if (x, y) in visited:
            continue

        visited.add((x, y))

        for nx, ny in env.get_neighbors(x, y):
            g = len(path)
            h = heuristic((nx, ny), goal)
            heapq.heappush(pq, (g + h, (nx, ny), path + [(nx, ny)]))

    return None

def a_star_risk(env, start, goal, risk_weight=3):
    pq = []
    heapq.heappush(pq, (0, start, [start]))

    visited = set()

    while pq:
        cost, (x, y), path = heapq.heappop(pq)

        if (x, y) == goal:
            return path

        if (x, y) in visited:
            continue

        visited.add((x, y))

        for nx, ny in env.get_neighbors(x, y):
            g = len(path)

            risk = env.grid[nx][ny].risk_level * risk_weight

            h = heuristic((nx, ny), goal)

            total_cost = g + h + risk

            heapq.heappush(pq, (total_cost, (nx, ny), path + [(nx, ny)]))

    return None

def greedy_best_first(env, start, goal):
    import heapq

    pq = []
    heapq.heappush(pq, (0, start, [start]))

    visited = set()

    while pq:
        h, (x, y), path = heapq.heappop(pq)

        if (x, y) == goal:
            return path

        if (x, y) in visited:
            continue

        visited.add((x, y))

        for nx, ny in env.get_neighbors(x, y):
            h = heuristic((nx, ny), goal)

            heapq.heappush(pq, (h, (nx, ny), path + [(nx, ny)]))

    return None

def hill_climbing(env, start, goal):
    current = start
    path = [current]

    while current != goal:
        x, y = current

        neighbors = env.get_neighbors(x, y)

        if not neighbors:
            return path  # stuck

        # choose best neighbor based on heuristic
        best = None
        best_h = float('inf')

        for nx, ny in neighbors:
            h = heuristic((nx, ny), goal)
            if h < best_h:
                best_h = h
                best = (nx, ny)

        # if no improvement -> stuck
        if best_h >= heuristic(current, goal):
            print("Hill Climbing stuck at:", current)
            return path

        current = best
        path.append(current)

    return path