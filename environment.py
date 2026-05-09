import random

# CELL (each position in grid)

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_blocked = False
        self.risk_level = 0  # 0 = safe, 1 = low, 2 = medium, 3 = high
        self.has_victim = False
        self.victim = None
        self.is_hospital = False
        self.is_base = False

# VICTIM

class Victim:
    def __init__(self, vid, x, y, severity):
        self.id = vid
        self.x = x
        self.y = y
        self.severity = severity  # critical, moderate, minor
        self.rescued = False

# ENVIRONMENT

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.grid = [[Cell(i, j) for j in range(size)] for i in range(size)]
        self.victims = []
        self.hospitals = []
        self.base = None

    # SETUP FUNCTIONS

    def set_base(self, x, y):
        self.grid[x][y].is_base = True
        self.base = (x, y)

    def add_hospital(self, x, y):
        self.grid[x][y].is_hospital = True
        self.hospitals.append((x, y))

    def add_victim(self, vid, x, y, severity):
        v = Victim(vid, x, y, severity)
        self.victims.append(v)
        self.grid[x][y].has_victim = True
        self.grid[x][y].victim = v

    def add_risk_zone(self, x, y, level):
        self.grid[x][y].risk_level = level

    def block_cell(self, x, y):
        self.grid[x][y].is_blocked = True


    # HELPER FUNCTIONS

    def is_valid(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and not self.grid[x][y].is_blocked

    def get_neighbors(self, x, y):
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        neighbors = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

    # DISPLAY GRID

    def display(self):
        print("\n--- ENVIRONMENT MAP ---\n")
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i][j]

                if cell.is_base:
                    print("B", end=" ")
                elif cell.is_hospital:
                    print("H", end=" ")
                elif cell.is_blocked:
                    print("X", end=" ")
                elif cell.has_victim:
                    print("V", end=" ")
                elif cell.risk_level > 0:
                    print("R", end=" ")
                else:
                    print(".", end=" ")
            print()

    def display_path(self, path):
        print("\n--- FULL RESCUE PATH VISUALIZATION ---\n")

        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.base:
                    print("B", end=" ")
                elif (i, j) in path:
                    print("*", end=" ")
                elif self.grid[i][j].is_base:
                    print("B", end=" ")
                elif self.grid[i][j].is_hospital:
                    print("H", end=" ")
                elif self.grid[i][j].is_blocked:
                    print("X", end=" ")
                elif self.grid[i][j].has_victim:
                    print("V", end=" ")
                elif self.grid[i][j].risk_level > 0:
                    print("R", end=" ")
                else:
                    print(".", end=" ")
            print()