import numpy as np
from matplotlib import pyplot as plt

n = 5
m = 2
C1 = 5
C2 = 11
NUM_STEPS = 10


def main():
    payoff_matrix = np.random.randint(
        low=C1,
        high=C2 + 1,
        size=(m, n),
    )
    # payoff_matrix = np.array([
    #     [7, 5, 8, 10, 7],
    #     [10, 8, 6, 10, 6],
    # ])
    p = np.random.random(size=(m,))
    p = p / np.sum(p)
    q = np.random.random(size=(n,))
    q = q / np.sum(q)

    print("Payoff matrix:")
    print(payoff_matrix, "\n")

    print("--- STEP #1")
    print(f"p = {p}")
    print(f"q = {q}")
    v_list = [0]

    for step in range(1, NUM_STEPS):
        p1_values, p2_values = play_game_round(
            payoff_matrix=payoff_matrix,
            step=step,
            p1_strategy_vector=p,
            p2_strategy_vector=q,
        )
        p1_idx, p1_gain, p = p1_values
        p2_idx, p2_loss, q = p2_values
        v = calculate_empirical_distribution(gain=p1_gain, loss=p2_loss)
        v_list.append(v)

        print(f"--- STEP #{step + 1}")
        print(f"\tα = {round(p1_gain, 2)}, i = {p1_idx}, p = {p}")
        print(f"\tβ = {round(p2_loss, 2)}, j = {p2_idx}, q = {q}")
        print(f"\tv = {round(v, 2)}")

    ax = plt.figure(figsize=(5, 3.5)).gca()
    ax.plot(range(1, len(v_list) + 1), v_list)
    ax.set_xlabel("i (iteration)")
    ax.set_ylabel("V (empirical distribution)")
    ax.set_title("Brown-Robinson method")
    ax.grid()
    plt.show()


def play_game_round(payoff_matrix: np.ndarray, step: int,
                    p1_strategy_vector: np.ndarray, p2_strategy_vector: np.ndarray):
    p1_idx, p1_gain = calculate_max_gain(
        payoff_matrix=payoff_matrix,
        opponent_probability_vector=p2_strategy_vector,
    )
    p2_idx, p2_loss = calculate_min_loss(
        payoff_matrix=payoff_matrix,
        opponent_probability_vector=p1_strategy_vector,
    )
    p1_strategy_vector = calculate_next_probability_vector(
        current_vector=p1_strategy_vector,
        current_step=step,
        next_idx=p1_idx,
    )
    p2_strategy_vector = calculate_next_probability_vector(
        current_vector=p2_strategy_vector,
        current_step=step,
        next_idx=p2_idx,
    )
    return (
        (p1_idx, p1_gain, p1_strategy_vector),
        (p2_idx, p2_loss, p2_strategy_vector),
    )


def calculate_empirical_distribution(gain: float, loss: float):
    return (gain + loss) / 2.


def calculate_min_loss(payoff_matrix: np.ndarray, opponent_probability_vector: np.ndarray):
    """Calculate best strategy to minimize loss of 2nd player using opponents' probability vector."""
    probable_loss = np.dot(payoff_matrix.T, opponent_probability_vector)
    strategy_idx = int(np.argmin(probable_loss))
    min_loss = probable_loss[strategy_idx]
    return strategy_idx, min_loss


def calculate_max_gain(payoff_matrix: np.ndarray, opponent_probability_vector: np.ndarray):
    """Calculate best strategy to maximize gain of 1st player using opponents' probability vector."""
    probable_gains = np.dot(payoff_matrix, opponent_probability_vector)
    strategy_idx = int(np.argmax(probable_gains))
    max_gain = probable_gains[strategy_idx]
    return strategy_idx, max_gain


def calculate_next_probability_vector(current_vector: np.ndarray, current_step: int, next_idx: int):
    next_vector = current_step * current_vector / (current_step + 1.)
    next_vector[next_idx] += 1. / (current_step + 1.)
    return next_vector


if __name__ == '__main__':
    main()
