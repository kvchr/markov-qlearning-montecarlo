% Theme 3, question C: Q-learning when the dynamics and the cost are unknown,
% and the effect of the discount factor, the learning rate and the exploration rate
rng(0);  % Fixed seed, so the results in the report can be reproduced

% Define parameters
states = 1:10;  % States 1 through 10
actions = [-1, 1];  % Actions: -1 (left), +1 (right)
g = [1, 2, 3, 4, 5, 4, 2, 0, 1, 2];  % Cost function

% Transition model, used only to simulate the system (the agent does not know it).
% Column 1: possible next states for u = -1, column 2: for u = +1 (same order as actions).
% Each next state has probability 0.5. States 1 and 10 are neighbours.
P = {
    [10, 1], [1, 2];   % State 1
    [1, 2],  [2, 3];   % State 2
    [2, 3],  [3, 4];   % State 3
    [3, 4],  [4, 5];   % State 4
    [4, 5],  [5, 6];   % State 5
    [5, 6],  [6, 7];   % State 6
    [6, 7],  [7, 8];   % State 7
    [7, 8],  [8, 9];   % State 8
    [8, 9],  [9, 10];  % State 9
    [9, 10], [10, 1];  % State 10
};

% Q-learning parameters (default values)
alpha = 0.2;  % Learning rate
gamma = 0.9;  % Discount factor (used while alpha or epsilon is varied)
epsilon = 0.1;  % Exploration rate
trajectory_length = 10;  % Steps per episode
hold_episodes = 100;  % Converged: the learned policy is optimal for this many episodes in a row
max_episodes = 50000;  % A run that has not converged by then is stopped
num_runs = 20;  % Independent runs per parameter value

% Values tested, one parameter at a time
gammas = linspace(0.1, 0.99, 10);  % Discount factors to test
alphas = [0.05, 0.1, 0.2, 0.4, 0.8];  % Learning rates to test
epsilons = [0.02, 0.05, 0.1, 0.2, 0.5];  % Exploration rates to test

% Experiment 1: discount factor (alpha = 0.2, epsilon = 0.1)
fprintf('Effect of the discount factor (alpha = %.2f, epsilon = %.2f)\n', alpha, epsilon);
E_gamma = zeros(num_runs, length(gammas));
for i = 1:length(gammas)
    E_gamma(:, i) = run_experiment(gammas(i), alpha, epsilon, states, actions, P, g, ...
        trajectory_length, hold_episodes, max_episodes, num_runs, sprintf('Gamma: %.2f', gammas(i)));
end

% Experiment 2: learning rate (gamma = 0.9, epsilon = 0.1)
fprintf('\nEffect of the learning rate (gamma = %.2f, epsilon = %.2f)\n', gamma, epsilon);
E_alpha = zeros(num_runs, length(alphas));
for i = 1:length(alphas)
    E_alpha(:, i) = run_experiment(gamma, alphas(i), epsilon, states, actions, P, g, ...
        trajectory_length, hold_episodes, max_episodes, num_runs, sprintf('Alpha: %.2f', alphas(i)));
end

% Experiment 3: exploration rate (gamma = 0.9, alpha = 0.2)
fprintf('\nEffect of the exploration rate (gamma = %.2f, alpha = %.2f)\n', gamma, alpha);
E_epsilon = zeros(num_runs, length(epsilons));
for i = 1:length(epsilons)
    E_epsilon(:, i) = run_experiment(gamma, alpha, epsilons(i), states, actions, P, g, ...
        trajectory_length, hold_episodes, max_episodes, num_runs, sprintf('Epsilon: %.2f', epsilons(i)));
end

% Plot results: median and 25th-75th percentile of the runs
figure('Position', [100 100 1400 430]);
subplot(1, 3, 1);
plot_median(gammas, E_gamma, 'Discount factor \gamma', max_episodes);
title(sprintf('\\alpha = %.1f, \\epsilon = %.1f', alpha, epsilon));
subplot(1, 3, 2);
plot_median(alphas, E_alpha, 'Learning rate \alpha', max_episodes);
set(gca, 'XScale', 'log');
title(sprintf('\\gamma = %.1f, \\epsilon = %.1f', gamma, epsilon));
subplot(1, 3, 3);
plot_median(epsilons, E_epsilon, 'Exploration rate \epsilon', max_episodes);
set(gca, 'XScale', 'log');
title(sprintf('\\gamma = %.1f, \\alpha = %.1f', gamma, alpha));

% Optimal Q-values by value iteration. Used only to check convergence; the agent never sees them.
function Q_star = optimal_q(gamma, P, g)
    Q_star = zeros(10, 2);
    for iter = 1:100000
        V = min(Q_star, [], 2);
        Q_new = zeros(10, 2);
        for s = 1:10
            for a = 1:2
                Q_new(s, a) = g(s) + gamma * mean(V(P{s, a}));  % Each next state has probability 0.5
            end
        end
        change = max(abs(Q_new(:) - Q_star(:)));
        Q_star = Q_new;
        if change < 1e-10
            break;
        end
    end
end

% Q-learning algorithm
function [num_episodes, Q, converged] = q_learning(gamma, states, actions, P, g, alpha, epsilon, trajectory_length, best_policy, hold_episodes, max_episodes)
    % Initialize Q-table
    Q = zeros(length(states), length(actions));
    streak = 0;  % Episodes in a row in which the greedy policy is the optimal one

    for num_episodes = 1:max_episodes
        state = randi(length(states));  % Start at a random state

        for t = 1:trajectory_length
            % Choose action using epsilon-greedy
            if rand < epsilon
                action = randi(length(actions));  % Random action
            else
                [~, action] = min(Q(state, :));  % Action minimizing Q-value
            end

            % Simulate transition for the chosen action
            next_state = P{state, action}(randi([1, 2]));  % Next state
            cost = g(state);  % Immediate cost

            % Update Q-value
            best_next_q = min(Q(next_state, :));
            Q(state, action) = Q(state, action) + alpha * (cost + gamma * best_next_q - Q(state, action));
            state = next_state;
        end

        % Check for convergence: is the greedy policy the optimal one?
        [~, greedy] = min(Q, [], 2);
        if isequal(greedy, best_policy)
            streak = streak + 1;
        else
            streak = 0;
        end
        if streak >= hold_episodes
            break;
        end
    end
    converged = streak >= hold_episodes;
    if converged
        num_episodes = num_episodes - hold_episodes + 1;  % Episode from which the policy stayed optimal
    end
end

% Run Q-learning num_runs times for one parameter setting and print a summary
function episodes = run_experiment(gamma, alpha, epsilon, states, actions, P, g, trajectory_length, hold_episodes, max_episodes, num_runs, label)
    Q_star = optimal_q(gamma, P, g);
    [~, best_policy] = min(Q_star, [], 2);
    episodes = zeros(num_runs, 1);
    not_converged = 0;
    for r = 1:num_runs
        [episodes(r), ~, converged] = q_learning(gamma, states, actions, P, g, alpha, epsilon, ...
            trajectory_length, best_policy, hold_episodes, max_episodes);
        not_converged = not_converged + ~converged;
    end
    s = sort(episodes);
    fprintf('%s  Optimal policy: %s  Median episodes: %d (25th-75th percentile: %d-%d)  Not converged after %d episodes: %d of %d runs\n', ...
        label, mat2str(actions(best_policy)), round(median(episodes)), s(round(0.25 * num_runs)), s(round(0.75 * num_runs)), ...
        max_episodes, not_converged, num_runs);
end

% Plot the median and the 25th-75th percentile of the episodes to convergence
function plot_median(x, E, x_label, max_episodes)
    s = sort(E, 1);
    n = size(E, 1);
    med = median(E, 1);
    errorbar(x, med, med - s(round(0.25 * n), :), s(round(0.75 * n), :) - med, '-o');
    yline(max_episodes, '--', 'limit (not converged)', 'LabelHorizontalAlignment', 'left');
    set(gca, 'YScale', 'log');
    ylim([100, 2 * max_episodes]);
    xlabel(x_label);
    ylabel('Episodes to convergence');
    grid on;
end
