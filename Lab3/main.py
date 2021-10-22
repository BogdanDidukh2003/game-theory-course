import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString

n = 5
m = 5
C1 = 3
C2 = 12


def main():
    payoff_matrix_2n = np.random.randint(
        low=C1,
        high=C2 + 1,
        size=(2, n),
    )
    payoff_matrix_m2 = np.random.randint(
        low=C1,
        high=C2 + 1,
        size=(m, 2),
    )
    payoff_matrix_2n = np.array([
        [7, 5, 8, 10, 7],
        [10, 8, 6, 9, 6],
    ])
    payoff_matrix_m2 = np.array([
        [6, 5],
        [4, 6],
        [3, 7],
        [9, 8],
        [3, 11],
    ])

    print("Payoff matrix 2xN:")
    print(payoff_matrix_2n)
    print("Payoff matrix Mx2:")
    print(payoff_matrix_m2)
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("Graphical method for mixed strategy game")

    solve_game_using_graphical_method(
        ax=ax1,
        payoff_matrix=payoff_matrix_2n,
    )
    solve_game_using_graphical_method(
        ax=ax2,
        payoff_matrix=payoff_matrix_m2,
    )
    plt.savefig("out.png")


def solve_game_using_graphical_method(ax, payoff_matrix):
    is_2n_game = True
    if payoff_matrix.shape[0] != 2:
        payoff_matrix = payoff_matrix.T
        is_2n_game = False
    # probability vector:
    # 0 -> (1, 0)
    # 1 -> (0, 1)
    # x -> (1-x, x)

    line_objects, line_functions = extract_line_properties(payoff_matrix)
    y_at_limits = []
    for line_fn in line_functions:
        y_at_limits.append(
            (line_fn(0), line_fn(1))
        )

    potential_result_x_set = calculate_potential_result_points_x(line_objects)
    if is_2n_game:
        strategy_optimal_coordinates = find_optimal_xy_for_2n_game(
            potential_result_x_set,
            line_functions,
        )
    else:
        strategy_optimal_coordinates = find_optimal_xy_for_m2_game(
            potential_result_x_set,
            line_functions,
        )
    optimal_x, optimal_y = strategy_optimal_coordinates
    p_strategy = 1 - optimal_x, optimal_x

    print("---", "2xN game" if is_2n_game else "Mx2 game")
    print(f"\tOptimal strategy: {'X' if is_2n_game else 'Y'}* "
          f"= ({p_strategy[0]:.2f}, {p_strategy[1]:.2f})")
    print(f"\tValue of the game: V = {optimal_y:.2f}")

    for i, y in enumerate(y_at_limits):
        ax.plot((0, 1), y, label=f'L{i + 1}')
    ax.plot((optimal_x, optimal_x), (0, optimal_y), "--k")
    ax.plot(optimal_x, optimal_y, ".k")
    if is_2n_game:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel("y")
        ax.set_ylabel("x")
    if is_2n_game:
        ax.set_title("2xN game")
    else:
        ax.set_title("Mx2 game")
    ax.set_xlim(left=-0.01, right=1.01)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(11) / 10.)
    ax.grid()
    ax.legend()


def find_optimal_xy_for_2n_game(potential_x_set, line_functions):
    min_values = []
    for x in potential_x_set:
        min_values.append(
            min([[x, line_fn(x)] for line_fn in line_functions], key=lambda i: i[1])
        )
    optimal_xy = max(min_values, key=lambda i: i[1])
    return optimal_xy


def find_optimal_xy_for_m2_game(potential_x_set, line_functions):
    max_values = []
    for x in potential_x_set:
        max_values.append(
            max([[x, line_fn(x)] for line_fn in line_functions], key=lambda i: i[1])
        )
    optimal_xy = min(max_values, key=lambda i: i[1])
    return optimal_xy


def calculate_potential_result_points_x(line_objects):
    """Find all `X`s of intersection points and points at limits of functions."""
    result_points_x = set()
    for line_obj_left in line_objects:
        for line_obj_right in line_objects:
            if line_obj_left == line_obj_right:
                continue
            intersection_point = line_obj_left.intersection(line_obj_right)
            if intersection_point:
                result_points_x.add(
                    intersection_point.x
                )
    result_points_x.add(0)
    result_points_x.add(1)
    return result_points_x


def extract_line_properties(payoff_matrix: np.ndarray):
    _, num_columns = payoff_matrix.shape
    line_objects, line_functions = [], []
    for j in range(num_columns):
        line_edges = payoff_matrix[:, j]
        line_obj = LineString(((0, line_edges[0]), (1, line_edges[1])))

        line_objects.append(line_obj)
        line_functions.append(
            lambda x, edges=line_edges: edges[0] * (1 - x) + edges[1] * x
        )
    return line_objects, line_functions


if __name__ == '__main__':
    main()
