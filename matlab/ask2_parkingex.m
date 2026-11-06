% Define parameters
N = 200;          % Number of probable parking spots
C = 100;          % Cost of parking at the garage
p = 0.05;         % Probability of a parking spot being empty

% Initialize cost array and decision array
expected_cost = zeros(1, N+1); % Include garage as N+1
decision = zeros(1, N); % Decision: 1 if park, 0 if skip

% Garage has a fixed cost
expected_cost(N+1) = C;

% Compute the optimal expected cost for each spot k in reverse order
% (Bellman equation: J(k) = p*min(c(k), J(k+1)) + (1-p)*J(k+1))
for k = N:-1:1
    % Cost of parking at spot k
    cost_if_park = N - k;

    % Expected cost if the driver drives on: the optimal expected cost from spot k+1
    cost_if_skip = expected_cost(k+1);

    % Spot k is free with probability p: then take the cheaper option.
    % Otherwise (probability 1-p) the driver has to drive on.
    expected_cost(k) = p * min(cost_if_park, cost_if_skip) + (1 - p) * cost_if_skip;

    % Optimal strategy: park at a free spot k if cost_if_park <= cost_if_skip
    decision(k) = cost_if_park <= cost_if_skip;
end

% Print the threshold
k_star = find(decision, 1);
fprintf('Park at the first free spot from k = %d on (price <= %d).\n', k_star, N - k_star);
fprintf('Optimal expected cost from the start: %.2f\n', expected_cost(1));

% Plot the expected cost for each spot
figure;
subplot(2, 1, 1);
plot(1:N, expected_cost(1:N), 'b-', 'LineWidth', 2);
hold on;
plot(N+1, expected_cost(N+1), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Parking Spot (k)');
ylabel('Expected Cost');
title('Optimal Expected Cost vs. Parking Spot');
grid on;
legend('Expected Cost', 'Garage Cost', 'Location', 'Best');

% Plot the decision strategy
subplot(2, 1, 2);
stem(1:N, decision, 'r', 'LineWidth', 1.5);
xlabel('Parking Spot (k)');
ylabel('Decision (1 = Park, 0 = Skip)');
title('Optimal Parking Strategy');
grid on;