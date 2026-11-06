function MarkovChain1_2
    % Theme 1, question B: state probabilities at t = 1000-1003 for every start state
    % Parameter sets and time steps
    param_sets = [0, 0.1, 0; 0.1, 0.1, 0.1];
    time_steps = [1000, 1001, 1002, 1003];
    num_states = 5;

    % Loop through parameter sets
    for idx = 1:size(param_sets, 1)
        a = param_sets(idx, 1);
        b = param_sets(idx, 2);
        c = param_sets(idx, 3);

        fprintf('Parameters: a=%.1f, b=%.1f, c=%.1f\n', a, b, c);
        P = markov_chain(a, b, c);
        validate_transition_matrix(P);

        % Compute matrix powers for time steps
        matrix_powers = cell(1, length(time_steps));
        for t_idx = 1:length(time_steps)
            matrix_powers{t_idx} = P^time_steps(t_idx);
        end

        % Calculate probabilities for each initial state
        for t_idx = 1:length(time_steps)
            fprintf('\nTime %d:\n', time_steps(t_idx));
            for start_state = 1:num_states
                initial_conditions = zeros(1, num_states);
                initial_conditions(start_state) = 1;
                probabilities = initial_conditions * matrix_powers{t_idx};
                fprintf('  Start state %d: %s\n', start_state, mat2str(probabilities, 6));
            end
        end
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
