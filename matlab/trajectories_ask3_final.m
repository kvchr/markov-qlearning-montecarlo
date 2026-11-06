% Theme 3, question B: simulated trajectories under the optimal controller
rng(0);  % Fixed seed, so the results in the report can be reproduced

% Define parameters
states = 1:10;  % States 1 through 10
g = [1, 2, 3, 4, 5, 4, 2, 0, 1, 2];  % Cost function

% Transition probabilities: for each state, row 1 is u = +1 and row 2 is u = -1
% (possible next states and their probabilities). States 1 and 10 are neighbours.
P = {
    {+1, [1, 2], [0.5, 0.5]; -1, [10, 1], [0.5, 0.5]},  % State 1
    {+1, [2, 3], [0.5, 0.5]; -1, [1, 2], [0.5, 0.5]},  % State 2
    {+1, [3, 4], [0.5, 0.5]; -1, [2, 3], [0.5, 0.5]},  % State 3
    {+1, [4, 5], [0.5, 0.5]; -1, [3, 4], [0.5, 0.5]},  % State 4
    {+1, [5, 6], [0.5, 0.5]; -1, [4, 5], [0.5, 0.5]},  % State 5
    {+1, [6, 7], [0.5, 0.5]; -1, [5, 6], [0.5, 0.5]},  % State 6
    {+1, [7, 8], [0.5, 0.5]; -1, [6, 7], [0.5, 0.5]},  % State 7
    {+1, [8, 9], [0.5, 0.5]; -1, [7, 8], [0.5, 0.5]},  % State 8
    {+1, [9, 10], [0.5, 0.5]; -1, [8, 9], [0.5, 0.5]}, % State 9
    {+1, [10, 1], [0.5, 0.5]; -1, [9, 10], [0.5, 0.5]} % State 10
};

% Main script
alpha_values = 0.1:0.1:0.9; % 0 < alpha < 1 (for alpha >= 1 the total cost is infinite)
results = struct;

for i = 1:length(alpha_values)
    alpha = alpha_values(i);
    V = value_iteration(alpha, states, g, P, 1e-6, 1000);
    policy = get_policy(alpha, V, states, g, P);
    results(i).alpha = alpha;
    results(i).V = V;
    results(i).policy = policy;
end

% Parameters for trajectory simulation
trajectory_length = 20; % Number of steps in trajectory
start_state = 1; % Starting state
trajectories = struct;

% Simulate trajectories for each alpha value
for i = 1:length(results)
    alpha = results(i).alpha;
    policy = results(i).policy;
    trajectory = simulate_trajectory(policy, start_state, trajectory_length, P);
    trajectories(i).alpha = alpha;
    trajectories(i).trajectory = trajectory;
end

% Display and visualize trajectories
for i = 1:length(trajectories)
    fprintf('Alpha: %.2f\n', trajectories(i).alpha);
    fprintf('Simulated Trajectory: %s\n', mat2str(trajectories(i).trajectory));
    % Plot trajectory
    figure;
    plot(1:trajectory_length, trajectories(i).trajectory, '-o');
    title(sprintf('Simulated Trajectory for Alpha = %.2f', trajectories(i).alpha));
    xlabel('Step');
    ylabel('State');
    grid on;
end

% Value iteration function
function V = value_iteration(alpha, states, g, P, tol, max_iter)
    V = zeros(1, length(states));
    for iter = 1:max_iter
        V_new = zeros(1, length(states));
        for s = 1:length(states)
            costs = zeros(1, 2); % For two actions: row 1 of P{s} is u=+1, row 2 is u=-1
            for a_idx = 1:2
                transitions = P{s}(a_idx, :);
                next_states = transitions{2};
                probs = transitions{3};
                for k = 1:length(next_states)
                    next_state = next_states(k);
                    prob = probs(k);
                    costs(a_idx) = costs(a_idx) + alpha * prob * V(next_state);
                end
                costs(a_idx) = costs(a_idx) + g(s); % Immediate cost
            end
            V_new(s) = min(costs); % Minimum cost
        end
        % Convergence check
        change = max(abs(V_new - V));
        V = V_new;
        if change < tol
            break;
        end
    end
end

% Get optimal policy
function policy = get_policy(alpha, V, states, g, P)
    policy = zeros(1, length(states));
    for s = 1:length(states)
        action_costs = zeros(1, 2);
        for a_idx = 1:2
            transitions = P{s}(a_idx, :);
            next_states = transitions{2};
            probs = transitions{3};
            for k = 1:length(next_states)
                next_state = next_states(k);
                prob = probs(k);
                action_costs(a_idx) = action_costs(a_idx) + alpha * prob * V(next_state);
            end
            action_costs(a_idx) = action_costs(a_idx) + g(s);
        end
        [~, idx] = min(action_costs);
        policy(s) = P{s}{idx, 1}; % Action label (+1 or -1) stored in that row of P
    end
end

% Simulate trajectory
function trajectory = simulate_trajectory(policy, start_state, num_steps, P)
    trajectory = zeros(1, num_steps); % Preallocate trajectory
    trajectory(1) = start_state;
    current_state = start_state;
    for t = 2:num_steps
        action = policy(current_state); % Get action from policy
        action_idx = find([P{current_state}{:, 1}] == action); % Row of P for this action
        transitions = P{current_state}(action_idx, :);
        next_states = transitions{2};
        probs = transitions{3};
        % Sample next state based on transition probabilities
        next_state = randsample(next_states, 1, true, probs);
        trajectory(t) = next_state;
        current_state = next_state; % Update current state
    end
end
