"""
Lab1 is made with following assumptions:
    - there are 2 players
    - player 1 chooses index `i`
    - player 2 chooses index `j`
"""
import numpy as np

n = 5
m = 2
C1 = 5
C2 = 11


def main():
    game_model = np.random.randint(
        low=C1,
        high=C2 + 1,
        size=(m, n),
    )
    # game_model = np.array([
    #     [5, 8, 9, 11, 11],
    #     [9, 7, 6, 8, 10],
    # ])  # no saddle points
    # game_model = np.array([
    #     [10, 7, 9, 11, 7],
    #     [10, 6, 9, 9, 8],
    # ])  # saddle point at (0,1)
    # game_model = np.array([
    #     [5, 9, 6, 7, 10],
    #     [5, 6, 7, 9, 9],
    # ])  # saddle points at (0,0) and (1,0)

    print(game_model)

    p1_idx_array, p1_gain = execute_player1_strategy_all_solutions(game_model)
    print(f"maximin: index i = {list(p1_idx_array)}")
    print(f"maximin: gain α = {p1_gain}")
    p2_idx_array, p2_gain = execute_player2_strategy_all_solutions(game_model)
    print(f"minimax: index j = {list(p2_idx_array)}")
    print(f"minimax: gain α = {p2_gain}")

    saddle_points = find_all_saddle_points(game_model)
    if saddle_points:
        print(f"All saddle points: {saddle_points}")
    else:
        print("There are no saddle points!")


def argmax_all(a: np.ndarray):
    return np.argwhere(a == np.max(a)).flatten()


def argmin_all(a: np.ndarray):
    return np.argwhere(a == np.min(a)).flatten()


def find_all_saddle_points(game_model: np.ndarray):
    p1_idx_array, p1_gain = execute_player1_strategy_all_solutions(game_model)
    p2_idx_array, p2_gain = execute_player2_strategy_all_solutions(game_model)
    saddle_points = []
    if p1_gain == p2_gain:
        saddle_points = [(i, j) for i in p1_idx_array for j in p2_idx_array]
    return saddle_points


def execute_player1_strategy_all_solutions(game_model: np.ndarray):
    """Find `i` index and gain for maximin strategy in (m,n) size matrix.
        Returns
        List of (max_min_idx_array, max_min_gain) where:
            max_min_idx_array: array of all indices for best strategy
            max_min_gain: best gain following the strategy
    """
    min_gain_per_row = np.min(game_model, axis=1)
    max_min_idx_array = argmax_all(min_gain_per_row)
    max_min_gain = min_gain_per_row[max_min_idx_array[0]]
    return max_min_idx_array, max_min_gain


def execute_player2_strategy_all_solutions(game_model: np.ndarray):
    """Find `j` index and gain for minimax strategy in (m,n) size matrix.
        Returns
        List of (min_max_idx_array, min_max_gain) where:
            min_max_idx_array: array of all indices for best strategy
            min_max_gain: best gain following the strategy
    """
    max_gain_per_row = np.max(game_model, axis=0)
    min_max_idx_array = argmin_all(max_gain_per_row)
    min_max_gain = max_gain_per_row[min_max_idx_array[0]]
    return min_max_idx_array, min_max_gain


def execute_player1_strategy(game_model: np.ndarray):
    """Find `i` index and gain for maximin strategy in (m,n) size matrix."""
    min_gain = np.min(game_model, axis=1)
    max_min_idx = np.argmax(min_gain)
    max_min_gain = min_gain[max_min_idx]
    return max_min_idx, max_min_gain


def execute_player2_strategy(game_model: np.ndarray):
    """Find `j` index and gain for minimax strategy in (m,n) size matrix."""
    max_gain = np.max(game_model, axis=0)
    min_max_idx = np.argmin(max_gain)
    min_max_gain = max_gain[min_max_idx]
    return min_max_idx, min_max_gain


if __name__ == '__main__':
    main()
