import numpy as np
import pulp as plp

n = 5
m = 2
C1 = 5
C2 = 11


def main():
    payoff_matrix = np.random.randint(
        low=C1,
        high=C2 + 1,
        size=(m, n),
    )
    payoff_matrix = np.array([
        [5, 8, 9, 11, 11],
        [9, 7, 6, 8, 10],
    ])

    print("\tPayoff matrix")
    print(payoff_matrix)
    normalization_shift, payoff_matrix = normalize_payoff_matrix(payoff_matrix)
    print("\n\tPayoff matrix normalized")
    print(payoff_matrix)

    print("\n\tSolution")
    x = solve_game_for_player_1(payoff_matrix)
    print(f"Player 1 problem solution: x = {np.round(x, 3)}")
    y = solve_game_for_player_2(payoff_matrix)
    print(f"Player 2 problem solution: y = {np.round(y, 3)}")

    game_value, p1_strategy_vector, p2_strategy_vector = calculate_game_metrics(
        x=x, y=y, normalization_shift=normalization_shift)
    print(f"Normalized game value: V = {game_value + normalization_shift}")
    print(f"Game value: V* = {game_value}")
    print(f"Player 1 strategy: p = {p1_strategy_vector}")
    print(f"Player 2 strategy: q = {p2_strategy_vector}")


def calculate_game_metrics(x: np.ndarray, y: np.ndarray, normalization_shift: int = 0):
    normalized_game_value = 1 / np.sum(x)
    p1_strategy_vector = np.round(normalized_game_value * x, 3)
    p2_strategy_vector = np.round(normalized_game_value * y, 3)
    game_value = normalized_game_value - normalization_shift
    return game_value, p1_strategy_vector, p2_strategy_vector


def solve_game_for_player_1(payoff_matrix: np.ndarray):
    num_strategies = payoff_matrix.shape[0]
    problem = plp.LpProblem("Player1_Problem", plp.LpMinimize)
    x_variables = [
        plp.LpVariable(f"x{i + 1}", lowBound=0)
        for i in range(num_strategies)
    ]
    problem += plp.lpSum(x_variables)
    for strategy in payoff_matrix.T:
        problem += plp.lpSum([
            strategy[i] * x_variables[i]
            for i in range(len(x_variables))
        ]) >= 1
    problem.solve(plp.PULP_CBC_CMD(msg=False))
    x_result = np.array([variable.varValue for variable in problem.variables()])
    return x_result


def solve_game_for_player_2(payoff_matrix: np.ndarray):
    num_strategies = payoff_matrix.shape[1]
    problem = plp.LpProblem('Player2_Problem', plp.LpMaximize)
    y_variables = [
        plp.LpVariable(f"y{i + 1}", lowBound=0)
        for i in range(num_strategies)
    ]
    problem += plp.lpSum(y_variables)
    for strategy in payoff_matrix:
        problem += plp.lpSum([
            strategy[i] * y_variables[i]
            for i in range(len(y_variables))
        ]) <= 1
    problem.solve(plp.PULP_CBC_CMD(msg=False))
    y_result = np.array([variable.varValue for variable in problem.variables()])
    return y_result


def normalize_payoff_matrix(payoff_matrix: np.ndarray):
    min_element = np.min(payoff_matrix)
    normalization_shift = 0
    if min_element <= 0:
        normalization_shift = -min_element + 1
        payoff_matrix = payoff_matrix + normalization_shift
    return normalization_shift, payoff_matrix


if __name__ == "__main__":
    main()
