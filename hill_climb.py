import sys
import nfl_strategy as nfl
import itertools
import qfl

_slant_37 = [
    [(-1, 4, False), (20, 3, False), (10, 2, False), (7, 2, False), (4, 2, False)],
    [(1, 3, False), (10, 2, False), (5, 2, False), (1, 4, False), (2, 2, False)],
    [(8, 2, False), (20, 3, False), (4, 2, False), (2, 2, False), (6, 2, False)],
]

_y_cross_80 = [
    [(0, 1, True), (-7, 4, False), (0, 1, False), (0, 1, True), (24, 2, False)],
    [(0, 1, False), (0, 1, True), (-6, 4, False), (18, 2, False), (13, 2, False)],
    [(19, 4, False), (0, 1, True), (22, 2, False), (47, 3, False), (0, 1, False)],
]

_z_corner_82 = [
    [(28, 3, False), (-9, 4, False), (47, 3, False), (0, 1, False), (0, 1, False)],
    [(30, 3, False), (0, 1, True), (8, 2, False), (-8, 4, False), (0, 1, False)],
    [(38, 3, False), (-9, 4, False), (0, 1, False), (0, 1, True), (0, 1, False)],
]

game_parameters = [
    ([_slant_37, _y_cross_80, _z_corner_82], [0.20, 0.05, 0.25, 0.1, 0.4]),
    ([_y_cross_80, _z_corner_82, _slant_37], [0.05, 0.20, 0.1, 0.25, 0.4]),
]


def hill_climb_search(model, time_limit, num_games):
    current_time_partitions = [
        (0, 3),
        (3, 4),
        (4, 80),  # Final boundary fixed at 80
    ]
    current_down_partitions = [
        (0, 2.5),
        (2.5, 6),
        (6, 10),  # Final boundary fixed at 10
    ]
    step_size = 0.5  # Initial step size

    best_time_partitions = current_time_partitions
    best_down_partitions = current_down_partitions
    best_score = model.simulate(qfl.q_learn(model, time_limit), num_games)
    i = 0  # Step counter

    while True:
        step = step_size * (0.95**i)  # Decaying step size
        print("step: " + str(i))
        i += 1  # Increment step counter

        # Generate neighbor solutions by moving the boundaries up and down by 'step'

        neighbors = []
        for partition_set, partition_type in [
            (current_time_partitions, "time"),
            (current_down_partitions, "down"),
        ]:
            for j in range(
                1, len(partition_set) - 1
            ):  # Exclude the fixed outer boundaries
                for boundary_index in [
                    0,
                    1,
                ]:  # 0 for the start, 1 for the end of each partition
                    for direction in [-1, 1]:
                        new_partitions = list(
                            partition_set
                        )  # Make a copy of the current partitions
                        # Adjust the selected boundary
                        new_bound = new_partitions[j][boundary_index] + direction * step
                        # Update the boundaries for the modified partition
                        if boundary_index == 0:
                            new_partitions[j - 1] = (
                                new_partitions[j - 1][0],
                                new_bound,
                            )
                            new_partitions[j] = (new_bound, new_partitions[j][1])
                        else:
                            new_partitions[j] = (new_partitions[j][0], new_bound)
                            if j < len(new_partitions) - 1:
                                new_partitions[j + 1] = (
                                    new_bound,
                                    new_partitions[j + 1][1],
                                )

                        neighbors.append((new_partitions, partition_type))

        for partitions, partition_type in neighbors:
            # Create new partition set for the evaluation
            time_partitions = (
                partitions if partition_type == "time" else current_time_partitions
            )
            down_partitions = (
                partitions if partition_type == "down" else current_down_partitions
            )

            score = model.simulate(
                qfl.q_learn(model, time_limit, time_partitions, down_partitions),
                num_games,
            )

            # If this neighbor is better, adopt it as the new best solution
            if score > best_score:
                best_score = score
                if partition_type == "time":
                    best_time_partitions = time_partitions
                    current_time_partitions = time_partitions
                else:
                    best_down_partitions = down_partitions
                    current_down_partitions = down_partitions

            print(time_partitions, down_partitions, score)

        # Termination condition: after a fixed number of steps
        if i >= 100:
            break

    return best_time_partitions, best_down_partitions, best_score


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "USAGE: {python3 | pypy3}",
            sys.argv[0],
            "model-number learning-time num-games",
        )
        sys.exit(1)

    try:
        game = int(sys.argv[1])
        limit = float(sys.argv[2])
        n = int(sys.argv[3])
    except:
        print(
            "USAGE: {python3 | pypy3}",
            sys.argv[0],
            "model-number learning-time num-games",
        )
        sys.exit(1)

    model = nfl.NFLStrategy(*game_parameters[game])
    print(hill_climb_search(model, limit, n))
