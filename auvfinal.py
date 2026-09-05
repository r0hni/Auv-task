
#UAV A* 

import heapq
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as matp
import numpy as np


# Grid making - 1

def make_grid(grid_line):
    grid = [row.split() for row in grid_line]
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        raise ValueError(f"Grid rows have inconsistent lengths: {widths}")
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == 'S':
                start = (r, c)
            elif ch == 'G':
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Grid must contain exactly one 'S' and one 'G'")
    return grid, start, goal


# Graph structure - 2


def struct(grid, cell):
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    moves = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (nr, nc) for nr, nc in moves
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#'
    ]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* search

def astar(grid, start, goal):
    counter = 0
    open_heap = [(manhattan(start, goal), counter, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {
                "path_found": True, "path": path, "cost": g_score[goal],
                "nodes_explored": len(closed), "explored_cells": closed,
            }

        for neighbor in struct(grid, current):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + manhattan(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return {
        "path_found": False, "path": None, "cost": None,
        "nodes_explored": len(closed), "explored_cells": closed,
    }



# Visualization

def visualize(grid, start, goal, result, title, save_path):
    rows, cols = len(grid), len(grid[0])
    img = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '#':
                img[r][c] = 1

    fig, ax = matp.subplots(figsize=(max(cols * 0.6, 3), max(rows * 0.6, 3)))
    ax.imshow(img, cmap='Greys', vmin=0, vmax=1)

    for (r, c) in result["explored_cells"]:
        if grid[r][c] != '#':
            ax.add_patch(matp.Rectangle((c - 0.5, r - 0.5), 1, 1, color='#cfe8ff', zorder=1))

    if result["path_found"]:
        path = result["path"]
        for (r, c) in path:
            ax.add_patch(matp.Rectangle((c - 0.5, r - 0.5), 1, 1, color='#ffd966', zorder=2))
        ax.plot([p[1] for p in path], [p[0] for p in path], color='orange', linewidth=2, zorder=3)

    sr, sc = start
    gr, gc = goal
    ax.text(sc, sr, 'S', ha='center', va='center', fontweight='bold', color='green', fontsize=14, zorder=4)
    ax.text(gc, gr, 'G', ha='center', va='center', fontweight='bold', color='red', fontsize=14, zorder=4)

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    matp.tight_layout()
    matp.savefig(save_path, dpi=120)
    matp.close(fig)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
TEST_CASES = {
    "Test Case 1: Simple Path": [
        "S . . . . . . .",
        ". . # # # . . .",
        ". . . . # . . .",
        ". # . . . # . .",
        ". # . . . . . .",
        ". . . . . . . G",
    ],
    "Test Case 2: Multiple Possible Paths": [
        "S . . . . . .",
        ". . . . . . .",
        ". . . # . . .",
        ". . . # . . .",
        ". . . # # . .",
        ". . . . . . G",
    ],
    "Test Case 3: Narrow Passage": [
        "S # . . . . .",
        ". # . # # # .",
        ". # . # . . .",
        ". # . # . # .",
        ". . . # . # .",
        "# # # # . # .",
        ". . . . . # G",
    ],
    "Test Case 4: Different Obstacle Arrangement": [
        "S . # . . . .",
        ". . # . # . .",
        ". . # . # . .",
        ". . . . # . .",
        "# # # . # . .",
        ". . . . # . .",
        ". # # . . . G",
    ],
    "Test Case 5: No Valid Path": [
        "S # . . . .",
        "# # . . . .",
        ". . . . . .",
        ". . . . . .",
        ". . . . # #",
        ". . . . # G",
    ],
}


def run_case(name, grid_lines, log_lines):
    grid, start, goal = make_grid(grid_lines)
    t0 = time.perf_counter()
    result = astar(grid, start, goal)
    elapsed = time.perf_counter() - t0

    log_lines.append("-----------Search Result-----------")
    log_lines.append(f"{name} Path Found: {'YES' if result['path_found'] else 'NO'}")
    if result["path_found"]:
        log_lines.append("Path:")
        for cell in result["path"]:
            log_lines.append(f"{cell}")
        log_lines.append(f"Total Path Cost: {result['cost']}  Nodes Explored: {result['nodes_explored']}")
    else:
        log_lines.append(f"No valid path exists between Start and Goal. Nodes Explored: {result['nodes_explored']}")

    fname = f"{name.replace(' ', '_').replace(':', '')}.png"
    visualize(grid, start, goal, result, name, fname)
    log_lines.append(f"Execution Time: {elapsed:.6f} seconds  Visualization saved: {fname}")
    log_lines.append("------------------------------------------\n")
    return result, elapsed


if __name__ == "__main__":
    log_lines = []
    for name, grid in TEST_CASES.items():
        run_case(name, grid, log_lines)
    log_text = "\n".join(log_lines)
    print(log_text)
    with open("simulation_log.txt", "w") as f:
        f.write(log_text)