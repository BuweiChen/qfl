# ~~~~~~~grid search optimization~~~~~~~
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

    def grid_search(model, time_limit):
        # Define the ranges for boundary values for each partition
        time_boundaries = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
        down_boundaries = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]

        # The best boundaries and their corresponding best score
        best_time_partitions = None
        best_down_partitions = None
        best_score = float("-inf")

        # Generate all combinations of the grid search
        for time_combination in itertools.combinations(time_boundaries, 2):
            for down_combination in itertools.combinations(down_boundaries, 2):
                # Create partitions from the boundary combinations
                time_partitions = [
                    (0, time_combination[0]),
                    (time_combination[0], time_combination[1]),
                    (time_combination[1], 80),
                ]
                down_partitions = [
                    (0, down_combination[0]),
                    (down_combination[0], down_combination[1]),
                    (down_combination[1], 10),
                ]

                # Train and evaluate the model with these partitions
                policy = qfl.q_learn(
                    model, time_limit, time_partitions, down_partitions
                )
                score = model.simulate(policy, n)

                # Update the best score and partitions
                if score > best_score:
                    best_score = score
                    best_time_partitions = time_partitions
                    best_down_partitions = down_partitions
                print(time_partitions, down_partitions, score)

        # Return the best found partitions and their score
        return best_time_partitions, best_down_partitions, best_score

    print(grid_search(model, limit))
