from math import factorial

V = {  # characteristic function coalition values of 3 players
    '': 0,
    '1': 40,
    '2': 60,
    '3': 100,
    '12': 120,
    '13': 170,
    '23': 200,
    '123': 300,
}
NUM_PLAYERS = 3


def main():
    print("\tSuper-additivity")
    additivity_list = []
    for players in ['12', '12', '23']:
        if len(players) > 1:
            is_super_additive = V[players] >= sum([V[player] for player in players])
            additivity_list.append(is_super_additive)

            print_sum_formula = " + ".join(["V(" + player + ")" for player in players])
            print(f"V({players}) >= {print_sum_formula}", end="")
            print("\t->\t", end="")
            print_sum_formula = " + ".join([str(V[player]) for player in players])
            sign = ">=" if is_super_additive else "<"
            print(f"{V[players]} {sign} {print_sum_formula}")

    if len(additivity_list) and all(additivity_list):
        print("Game is super additive")
    else:
        print("Game is not super additive")

    print("\n\tSub-additivity")
    players = ''.join(list(map(lambda x: str(x), range(1, NUM_PLAYERS + 1))))
    is_sub_additive = V[players] < sum([V[player] for player in players])

    print_sum_formula = " + ".join(["V(" + player + ")" for player in players])
    print(f"V({players}) > {print_sum_formula}", end="")
    print("\t->\t", end="")
    sign = "<=" if is_sub_additive else ">"
    print_sum_formula = " + ".join([str(V[player]) for player in players])
    print(f"{V[players]} {sign} {print_sum_formula}")

    if is_sub_additive:
        print("Game is sub additive")
    else:
        print("Game is not sub additive")

    print("\n\t0-1 Normalized Form")
    players = ''.join(list(map(lambda x: str(x), range(1, NUM_PLAYERS + 1))))

    normalized_v = dict()
    normalized_v[''] = 0
    for player in players:
        normalized_v[player] = 0

    def sum_values_of(group: str):
        return sum([V[player] for player in group])

    for group in ['12', '13', '23']:
        normalized_v[group] = (V[group] - sum_values_of(group)) / (V[players] - sum_values_of(players))
    normalized_v[players] = 1
    for k in normalized_v.keys():
        print(f"V'({','.join(k)}) = {normalized_v[k]:.2f}")

    print("\n\tCore")
    is_core_check_list = []

    def check_is_core(group: str):
        return normalized_v[group] <= 1 / (NUM_PLAYERS - len(group) + 1.)

    for k in normalized_v.keys():
        if not k:
            continue
        is_core = check_is_core(k)
        is_core_check_list.append(is_core)
        sign = "<=" if is_core else ">"
        print(f"V'({','.join(k)}) = {normalized_v[k]:.2f} {sign} {1 / (NUM_PLAYERS - len(k) + 1.):.2f}")

    if len(is_core_check_list) and all(is_core_check_list):
        print("Core is not empty")
    else:
        print("Core is empty")

    print("\n\tShapley values")

    def calculate_shapley_value(player: str):
        coalition_set = set(filter(lambda x, p=player: p in x, V))
        shapley_value = 0
        for coalition in coalition_set:
            shapley_value += (factorial(len(coalition) - 1) * factorial(NUM_PLAYERS - len(coalition))
                              ) / (factorial(NUM_PLAYERS)) * (
                                     V[coalition] - V[coalition.replace(player, "")])
        return shapley_value

    players = ''.join(list(map(lambda x: str(x), range(1, NUM_PLAYERS + 1))))
    shapley_vector = []
    for player in players:
        shapley_value = calculate_shapley_value(player)
        shapley_vector.append(shapley_value)
        print(f"φ_{player}(V) = {shapley_value:0.1f}")
    print()

    def check_is_core(group: str):
        # x1 = shapley_vector[int('1') - 1]
        return V[group] <= sum([shapley_vector[int(p) - 1] for p in group])

    is_core_check_list = []
    for group in V.keys():
        if not group:
            continue
        is_core = check_is_core(group)
        is_core_check_list.append(is_core)
        sign = "<=" if is_core else ">"
        print(f"V({','.join(group)}) = {V[group]} {sign} {sum([shapley_vector[int(p) - 1] for p in group]):.2f}")

    if len(is_core_check_list) and all(is_core_check_list):
        print("Shapley vector belongs to the core")
    else:
        print("Shapley vector doesn't belong to the core")


if __name__ == "__main__":
    main()
