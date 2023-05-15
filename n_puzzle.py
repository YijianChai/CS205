import heapq
import numpy as np

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




goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

test_cases = [
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]),
    np.array([[1, 2, 3], [4, 5, 6], [0, 7, 8]]),
    np.array([[1, 2, 3], [5, 0, 6], [4, 7, 8]]),
    np.array([[1, 3, 6], [5, 0, 2], [4, 7, 8]]),
    np.array([[1, 3, 6], [5, 0, 7], [4, 8, 2]]),
    np.array([[1, 6, 7], [5, 0, 3], [4, 8, 2]]),
    np.array([[7, 1, 2], [4, 8, 5], [6, 3, 0]]),
    np.array([[0, 7, 2], [4, 6, 1], [3, 5, 8]])
]


def main():
    print("Welcome to my n-Puzzle Solver. Type '1' to use a default puzzle, or '2' to create your own.")
    choice = int(input())
    if choice == 1:
        print("Choose one of the eight default puzzles (Type a number from 1 to 8):")
        for i, test_case in enumerate(test_cases, start=1):
            print(f"Test example {i}:\n {test_case}")
        puzzle_choice = int(input())
        n = 3
        puzzle = test_cases[puzzle_choice - 1]
    elif choice == 2:
        print("Enter the size of the puzzle (e.g., '3' for 8-puzzle, '4' for 15-puzzle, '5' for 24-puzzle, etc.)")
        n = int(input())
        puzzle = []
        print("Enter your puzzle, using a zero to represent the blank. Please only enter valid n-puzzles. \n"
              "Enter the puzzle demilimiting the numbers with a space. Type RETURN only when finished.")
        for i in range(n):
            row = list(map(int, input(f"Enter the {ordinal(i + 1)} row: ").split()))
            puzzle.append(row)

    print("Select algorithm. (1) for Uniform Cost Search, (2) for the Misplaced Tile Heuristic, or (3) the Manhattan Distance Heuristic.")
    algorithm_choice = int(input())

    # Generate the goal array based on different n
    goal = np.array(list(range(1, n * n)) + [0]).reshape(n, n)

    if algorithm_choice == 1:
        solution, depth, nodes_expanded, max_queue_size = uniform_cost_search(puzzle, goal, n)
    elif algorithm_choice == 2:
        solution, depth, nodes_expanded, max_queue_size = MTH(puzzle, goal, n)
    elif algorithm_choice == 3:
        solution, depth, nodes_expanded, max_queue_size = MDH(puzzle, goal, n)

    print("Solution:")
    for state in solution:
        print(state)
    print(f"Solution depth: {depth}")
    print(f"Number of nodes expanded: {nodes_expanded}")
    print(f"Max queue size: {max_queue_size}")

# Determine the ordinal of the given row, as we need to consider more complex conditions, such as 24-puzzle, 35-puzzle, etc.
def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix


if __name__ == "__main__":
    main()
