import heapq
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def neighbors(puzzle, n):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    empty_position = tuple(np.argwhere(puzzle == 0)[0])
    # Given the previous position of empty block, we can get every possible new positions.
    for move in moves:
        new_position = (empty_position[0] + move[0], empty_position[1] + move[1])
        # Valid range
        if (0 <= new_position[0] < n and 0 <= new_position[1] < n):
            new_puzzle = puzzle.copy()
            # Swap the positions
            new_puzzle[empty_position], new_puzzle[new_position] = new_puzzle[new_position], new_puzzle[empty_position]
            yield new_puzzle, 1

def uniform_cost_search(puzzle, goal, n):
    visited = set()
    queue = [(0, tuple(map(tuple, puzzle)), [])]
    nodes_expanded = 0
    max_queue_size = 1

    while queue:
        # In Uniform Cost Search f(n) = 0, so f(n) = g(n). We only need to consider the actual cost.
        (cost, current_state, path) = heapq.heappop(queue)
        current_state = np.array(current_state)
        # If we achieve our goal
        if np.array_equal(current_state, goal):
            return path, cost, nodes_expanded, max_queue_size

        # Convert the current state to hash code to test if it is visited.
        current_hash = current_state.tobytes()
        if current_hash not in visited:
            visited.add(current_hash)
            nodes_expanded += 1

            for neighbor, edge_cost in neighbors(current_state, n):
                neighbor_hash = neighbor.tobytes()
                # Only consider unvisited states
                if neighbor_hash not in visited:
                    heapq.heappush(queue, (cost + edge_cost, tuple(map(tuple, neighbor)), path + [tuple(map(tuple, neighbor))]))
                    max_queue_size = max(max_queue_size, len(queue))

    return None, None, None, None

# Approach 2:
def misplaced_tiles(puzzle, goal):
    return np.sum(puzzle != goal) - 1  # to exclude the blank tile

def MTH(puzzle, goal, n):
    visited = set()
    queue = [(0 + misplaced_tiles(puzzle, goal), 0, tuple(map(tuple, puzzle)), [])]
    nodes_expanded = 0
    max_queue_size = 1

    while queue:
        (priority, cost, current_state, path) = heapq.heappop(queue)
        current_state = np.array(current_state)
        # If we achieve our goal
        if np.array_equal(current_state, goal):
            return path, cost, nodes_expanded, max_queue_size

        # convert the current state to hash code to test if it is visited.
        current_hash = current_state.tobytes()
        if current_hash not in visited:
            visited.add(current_hash)
            nodes_expanded += 1

            for neighbor, edge_cost in neighbors(current_state, n):
                neighbor_hash = neighbor.tobytes()
                # Only consider unvisited states
                if neighbor_hash not in visited:
                    # f(n) = g(n) + h(n)
                    total_cost = cost + edge_cost + misplaced_tiles(neighbor, goal)
                    heapq.heappush(queue, (total_cost, cost + edge_cost, tuple(map(tuple, neighbor)), path + [tuple(map(tuple, neighbor))]))
                    max_queue_size = max(max_queue_size, len(queue))

    return None, None, None, None


# Approach 3
def manhattan_distance(puzzle, goal, n):
    distance = 0
    goal = np.array(goal)
    for i in range(n):
        for j in range(n):
            tile = puzzle[i][j]
            if tile != 0:
                goal_i, goal_j = np.where(goal == tile)
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def MDH(puzzle, goal, n):
    visited = set()
    queue = [(0 + manhattan_distance(puzzle, goal, n), 0, tuple(map(tuple, puzzle)), [])]
    nodes_expanded = 0
    max_queue_size = 1

    while queue:
        (priority, cost, current_state, path) = heapq.heappop(queue)
        current_state = np.array(current_state)
        # If we achieve our goal
        if np.array_equal(current_state, goal):
            return path, cost, nodes_expanded, max_queue_size

        # convert the current state to hash code to test if it is visited.
        current_hash = current_state.tobytes()
        if current_hash not in visited:
            visited.add(current_hash)
            nodes_expanded += 1

            # Only consider unvisited states
            for neighbor, edge_cost in neighbors(current_state, n):
                neighbor_hash = neighbor.tobytes()
                if neighbor_hash not in visited:
                    # f(n) = g(n) + h(n)
                    total_cost = cost + edge_cost + manhattan_distance(neighbor, goal, n)
                    heapq.heappush(queue, (total_cost, cost + edge_cost, tuple(map(tuple, neighbor)), path + [tuple(map(tuple, neighbor))]))
                    max_queue_size = max(max_queue_size, len(queue))

    return None, None, None, None

def generate_puzzle_with_depth(goal, depth):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Create a new array in order not to change the original "goal"
    new_puzzle = goal.copy()
    # Avoid to visit previous states
    visited = {new_puzzle.tobytes()}

    while depth > 0:
        empty_position = tuple(np.argwhere(new_puzzle == 0)[0])
        valid_moves = []

        for move in moves:
            new_pos = (empty_position[0] + move[0], empty_position[1] + move[1])
            if (0 <= new_pos[0] < 3 and 0 <= new_pos[1] < 3):
                valid_moves.append(move)

        selected_move = random.choice(valid_moves)
        new_pos = (empty_position[0] + selected_move[0], empty_position[1] + selected_move[1])
        new_puzzle[empty_position], new_puzzle[new_pos] = new_puzzle[new_pos], new_puzzle[empty_position]

        puzzle_hash = new_puzzle.tobytes()
        if puzzle_hash not in visited:
            visited.add(puzzle_hash)
            depth -= 1

    return new_puzzle




goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

def main():
    depth_range = list(range(1, 21))  # Change this range to the depth values you want to analyze
    num_trials = 100  # The number of trials for each depth

    ucs_nodes_expanded = [0] * len(depth_range)
    ucs_max_queue_size = [0] * len(depth_range)
    MTH_nodes_expanded = [0] * len(depth_range)
    MTH_max_queue_size = [0] * len(depth_range)
    MDH_nodes_expanded = [0] * len(depth_range)
    MDH_max_queue_size = [0] * len(depth_range)

    # To compute the average solving time of each algorithm
    ucs_solve_times = [0] * len(depth_range)
    MTH_solve_times = [0] * len(depth_range)
    MDH_solve_times = [0] * len(depth_range)

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}")
        for depth_idx, depth in enumerate(depth_range):
            print(f"Calculating performance for depth {depth}")

            puzzle = generate_puzzle_with_depth(goal, depth)
            # UCS
            start_time = time.time()
            _, _, ucs_expanded, ucs_queue = uniform_cost_search(puzzle, goal, 3)
            ucs_solve_times[depth_idx] += time.time() - start_time
            # MTH
            start_time = time.time()
            _, _, misplaced_expanded, misplaced_queue = MTH(puzzle, goal, 3)
            MTH_solve_times[depth_idx] += time.time() - start_time
            # MDH
            start_time = time.time()
            _, _, manhattan_expanded, manhattan_queue = MDH(puzzle, goal, 3)
            MDH_solve_times[depth_idx] += time.time() - start_time

            ucs_nodes_expanded[depth_idx] += ucs_expanded
            ucs_max_queue_size[depth_idx] += ucs_queue
            MTH_nodes_expanded[depth_idx] += misplaced_expanded
            MTH_max_queue_size[depth_idx] += misplaced_queue
            MDH_nodes_expanded[depth_idx] += manhattan_expanded
            MDH_max_queue_size[depth_idx] += manhattan_queue

    # Divide the accumulated values by the number of trials to get the average values
    ucs_nodes_expanded = [val / num_trials for val in ucs_nodes_expanded]
    ucs_max_queue_size = [val / num_trials for val in ucs_max_queue_size]
    MTH_nodes_expanded = [val / num_trials for val in MTH_nodes_expanded]
    MTH_max_queue_size = [val / num_trials for val in MTH_max_queue_size]
    MDH_nodes_expanded = [val / num_trials for val in MDH_nodes_expanded]
    MDH_max_queue_size = [val / num_trials for val in MDH_max_queue_size]
    ucs_solve_times = [val / num_trials for val in ucs_solve_times]
    MTH_solve_times = [val / num_trials for val in MTH_solve_times]
    MDH_solve_times = [val / num_trials for val in MDH_solve_times]

    # Figure 1: Plot the number of nodes expanded for each algorithm
    plt.plot(depth_range[:len(ucs_nodes_expanded)], ucs_nodes_expanded, label="Uniform Cost Search")
    plt.plot(depth_range[:len(MTH_nodes_expanded)], MTH_nodes_expanded, label="A* Misplaced Tiles")
    plt.plot(depth_range[:len(MDH_nodes_expanded)], MDH_nodes_expanded, label="A* Manhattan Distance")
    plt.title("Nodes Expanded vs Solution Depth")
    plt.xlabel("Depth")
    plt.ylabel("Number of Nodes Expanded")
    plt.xticks(depth_range)
    plt.legend()
    plt.show()

    # Figure 2: Plot the max queue size for each algorithm
    plt.plot(depth_range[:len(ucs_max_queue_size)], ucs_max_queue_size, label="Uniform Cost Search")
    plt.plot(depth_range[:len(MTH_max_queue_size)], MTH_max_queue_size, label="A* Misplaced Tiles")
    plt.plot(depth_range[:len(MDH_max_queue_size)], MDH_max_queue_size, label="A* Manhattan Distance")
    plt.title("Max Queue Size vs Solution Depth")
    plt.xlabel("Depth")
    plt.ylabel("Max Queue Size")
    plt.xticks(depth_range)
    plt.legend()
    plt.show()

    # Figure 3: Plot the average time for solving puzzle
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # First subplot for depth 0-10
    width = 0.3
    x1 = np.arange(11)
    axes[0].bar(x1 - width, ucs_solve_times[:11], width, label="Uniform Cost Search")
    axes[0].bar(x1, MTH_solve_times[:11], width, label="A* Misplaced Tiles")
    axes[0].bar(x1 + width, MDH_solve_times[:11], width, label="A* Manhattan Distance")

    axes[0].set_title("Average Solve Time vs Solution Depth (0-10)")
    axes[0].set_xlabel("Depth")
    axes[0].set_ylabel("Average Solve Time (seconds)")
    axes[0].set_xticks(x1)
    axes[0].legend()

    # Second subplot for depth 11-20
    x2 = np.arange(len(ucs_solve_times[11:])) + 11

    axes[1].bar(x2 - width, ucs_solve_times[11:], width, label="Uniform Cost Search")
    axes[1].bar(x2, MTH_solve_times[11:], width, label="A* Misplaced Tiles")
    axes[1].bar(x2 + width, MDH_solve_times[11:], width, label="A* Manhattan Distance")

    axes[1].set_title("Average Solve Time vs Solution Depth (11-20)")
    axes[1].set_xlabel("Depth")
    axes[1].set_ylabel("Average Solve Time (seconds)")
    axes[1].set_xticks(x2)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# Determine the ordinal of the given row, as we need to consider more complex conditions, such as 24-puzzle, 35-puzzle, etc.
def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix


if __name__ == "__main__":
    main()
