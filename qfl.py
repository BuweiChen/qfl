import random
import time


def q_learn(model, time_limit, time_partitions=None, down_partitions=None):
    # Constants
    gamma = 0.99  # discount factor
    epsilon = 0.15  # exploration rate
    if not time_partitions:
        time_partitions = [
            (0, 1.941989952767129),
            (1.941989952767129, 4),
            (4, 80),
        ]  # Partition bounds for yards per time tick
    if not down_partitions:
        down_partitions = [
            (0, 3.882253125),
            (3.882253125, 5.980250453046781),
            (5.980250453046781, 10),
        ]  # Partition bounds for yards per down

    # Initialize Q-values and visitation counts for partitions
    Q_values = {}
    visit_counts = {}
    for tp in range(len(time_partitions)):
        for dp in range(len(down_partitions)):
            for action in range(model.offensive_playbook_size()):
                Q_values[(tp, dp, action)] = 0
                visit_counts[(tp, dp, action)] = 0

    def get_partition(value, partitions):
        for i, (start, end) in enumerate(partitions):
            if start <= value < end:
                return i
        return len(partitions) - 1

    def extract_features(position):
        remaining_yards, downs_left, yards_for_down, remaining_time_ticks = position

        # Determine the partitions for each feature
        time_partition = get_partition(
            remaining_yards / remaining_time_ticks if remaining_time_ticks > 0 else 80,
            time_partitions,
        )
        down_partition = get_partition(
            yards_for_down / downs_left if downs_left > 0 else 10, down_partitions
        )

        return time_partition, down_partition

    def epsilon_greedy_action(state_partition, epsilon):
        if random.random() < epsilon:
            return random.randint(0, model.offensive_playbook_size() - 1)
        else:
            # Find the action with the highest Q-value for the state partition
            q_values = [
                Q_values[(state_partition + (a,))]
                for a in range(model.offensive_playbook_size())
            ]
            return max(range(len(q_values)), key=q_values.__getitem__)

    def collapsed_epsilon_greedy_action(state_partition):
        q_values = [
            Q_values[(state_partition + (a,))]
            for a in range(model.offensive_playbook_size())
        ]
        return max(range(len(q_values)), key=q_values.__getitem__)

    def update_q_value(state_partition, action, reward, next_state_partition, alpha):
        # Calculate the best next action Q-value
        next_max_q = max(
            Q_values[(next_state_partition + (a,))]
            for a in range(model.offensive_playbook_size())
        )
        # Q-learning update rule
        Q_values[(state_partition + (action,))] += alpha * (
            reward + gamma * next_max_q - Q_values[(state_partition + (action,))]
        )

    # Deterministic variation of policy
    def collapsed_policy(position):
        # Extract features and determine the current state partition
        state_partition = extract_features(position)
        # Select an action using epsilon-greedy strategy
        action = collapsed_epsilon_greedy_action(state_partition)
        return action

    # Policy function to be used for action selection
    def policy(position):
        # Extract features and determine the current state partition
        state_partition = extract_features(position)
        # Select an action using epsilon-greedy strategy
        action = epsilon_greedy_action(state_partition, epsilon)
        return action

    start_time = time.time()
    while time.time() - start_time < time_limit:
        position = model.initial_position()
        while not model.game_over(position):
            state_partition = extract_features(position)
            action = policy(position)
            new_position, outcome = model.result(position, action)
            next_state_partition = extract_features(new_position)
            # Determine the reward based on the new position and outcome
            if model.game_over(new_position):  # Check if the position is terminal
                if new_position[0] == 0:  # Check if touchdown was scored
                    reward = 1
                else:
                    reward = -1
            elif outcome[2]:  # Check if a turnover occurred
                reward = -1
            else:
                reward = outcome[0] * 0.0125

            # Update visitation counts and alpha
            visit_counts[(state_partition + (action,))] += 1
            alpha = 1 / visit_counts[(state_partition + (action,))]

            # Update the Q-value for the taken action
            update_q_value(state_partition, action, reward, next_state_partition, alpha)

            # Update the position
            position = new_position

    return collapsed_policy
