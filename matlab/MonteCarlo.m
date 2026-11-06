% Monte Carlo Simulation for Pi Estimation in MATLAB (Theme 4)

function MonteCarlo()
    rng(0);  % Fixed seed, so the results in the report can be reproduced

    % Parameters
    num_steps_list = [100, 500, 1000, 5000, 10000,50000]; % Number of points to test
    repetitions = 10; % Number of repetitions for variance evaluation

    % Run simulation
    results = simulate_and_evaluate(num_steps_list, repetitions);

    % Display results, next to the theoretical variance pi*(4-pi)/N
    fprintf('Points\tMean Pi\tVariance\tTheory\n');
    for i = 1:length(num_steps_list)
        num_points = num_steps_list(i);
        fprintf('%d\t%.5f\t%.5e\t%.5e\n', num_points, results(i).mean, results(i).variance, pi * (4 - pi) / num_points);
    end

    % Plot convergence
    means = [results.mean];
    variances = [results.variance];

    figure;
    subplot(1, 2, 1);
    semilogx(num_steps_list, means, '-o');
    title('Convergence of \pi Estimate');
    xlabel('Number of Points (log scale)');
    ylabel('Mean \pi Estimate');

    subplot(1, 2, 2);
    semilogx(num_steps_list, variances, '-o', 'Color', [0.8500, 0.3250, 0.0980]);
    title('Variance of \pi Estimate');
    xlabel('Number of Points (log scale)');
    ylabel('Variance');
end

function pi_estimate = monte_carlo_pi(num_points)
    % Generate random points
    x = rand(num_points, 1);
    y = rand(num_points, 1);

    % Check if points are inside the circle
    inside_circle = (x.^2 + y.^2) <= 1;

    % Estimate pi
    pi_estimate = 4 * sum(inside_circle) / num_points;
end

function results = simulate_and_evaluate(num_steps_list, repetitions)
    % Simulate Monte Carlo method for different point counts
    results = struct('mean', [], 'variance', []);

    for i = 1:length(num_steps_list)
        num_points = num_steps_list(i);
        estimates = zeros(repetitions, 1);

        for j = 1:repetitions
            estimates(j) = monte_carlo_pi(num_points);
        end

        % Store mean and variance
        results(i).mean = mean(estimates);
        results(i).variance = var(estimates);
    end
end