function MarkovChain1_3
    % Theme 1, question C: simulated trajectory vs stationary distribution
    rng(0);  % Fixed seed, so the results in the report can be reproduced

    % Parameters
    a = 0.1; b = 0.1; c = 0.1;
    num_steps = 10000; % Total steps
    num_states = 5;    % Number of states

    % Transition matrix
    P = markov_chain(a, b, c);
    validate_transition_matrix(P);

    % Initialize trajectory
    trajectory = zeros(1, num_steps);
    trajectory(1) = 1; % Start in state 1

    % Precompute cumulative probabilities
    cumulative_P = cumsum(P, 2);

    % Simulate trajectory
    for t = 2:num_steps
        current_state = trajectory(t - 1);
        rand_val = rand();
        trajectory(t) = find(rand_val <= cumulative_P(current_state, :), 1);
    end

    % Display last 100 states (5 rows of 20)
    fprintf('\nLast 100 states of the trajectory:\n');
    last_states = trajectory(end-99:end);
    for row = 1:5
        fprintf('%s\n', sprintf('%3d', last_states((row - 1) * 20 + (1:20))));
    end

    % Calculate time percentages for each state
    time_percentage = histcounts(trajectory, 1:(num_states + 1)) / num_steps * 100;
    fprintf('\nTime spent in each state:\n');
    for state = 1:num_states
        fprintf('State %d: %.2f%%\n', state, time_percentage(state));
    end

    % Calculate stationary distribution
    [stationary_vector, ~] = calculate_stationary_distribution(P);
    fprintf('\nStationary distribution:\n');
    for state = 1:num_states
        fprintf('State %d: %.3f\n', state, stationary_vector(state));
    end

    % Compare simulation with stationary distribution
    fprintf('\nComparison:\n');
    for state = 1:num_states
        fprintf('State %d: Simulated=%.2f%%, Stationary=%.2f%%\n', ...
            state, time_percentage(state), stationary_vector(state) * 100);
    end
end

function P = markov_chain(a, b, c)
    % Transition matrix
    P = [
        0, 1 - a - b, a, b, 0;
        0, 0, 1, 0, 0;
        0, 0, 0, 1 - c, c;
        1, 0, 0, 0, 0;
        0, 0, 0.5, 0, 0.5
    ];
end

function validate_transition_matrix(P)
    % Check if rows sum to 1
    for row = 1:size(P, 1)
        assert(abs(sum(P(row, :)) - 1) < 1e-6, 'Row %d does not sum to 1.', row);
    end
    fprintf('Transition matrix is valid.\n');
end

function [stationary_vector, eigenvalues] = calculate_stationary_distribution(P)
    % Compute stationary distribution
    [eigvecs, eigvals] = eig(P');
    eigenvalues = diag(eigvals);
    stationary_vector = eigvecs(:, abs(eigenvalues - 1) < 1e-6);
    stationary_vector = stationary_vector / sum(stationary_vector); % Normalize
    stationary_vector = abs(stationary_vector); % Ensure non-negative
end
