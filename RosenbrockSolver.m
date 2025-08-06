% rosenbrock_mdo.m
% Solve the Rosenbrock function via unconstrained optimization,
% record the solver path, and visualize results.

clear; clc; close all;

% 1. Initial guess
x0 = [-2; 3];

% 2. Set up global storage for history
global hist;
hist = [];  

% 3. Define the Rosenbrock function
rosen = @(x) (1 - x(1))^2 + 100*(x(2) - x(1)^2)^2;

% 4. Configure optimizer options with callback
options = optimoptions( ...
    'fminunc', ...
    'Algorithm',      'quasi-newton', ...   % gradient‐based
    'OutputFcn',      @outfun, ...          % record each iterate
    'Display',        'iter' ...            % show progress
);

% 5. Run the solver
[x_opt, f_opt, exitflag, output] = fminunc(rosen, x0, options);

% 6. Retrieve recorded history
history = hist.';   % each column of hist is an [x; y] pair

% 7. Display results
fprintf('Rosenbrock Function Minimum:\n');
fprintf('  Optimal x: %.4f\n', x_opt(1));
fprintf('  Optimal y: %.4f\n', x_opt(2));
fprintf('  Minimum f(x,y): %.4e\n', f_opt);

% 8. Prepare grid for plotting
x_plot = linspace(-2, 2, 400);
y_plot = linspace(-1, 3, 400);
[X, Y] = meshgrid(x_plot, y_plot);
Z = (1 - X).^2 + 100*(Y - X.^2).^2;

% 9. Plotting
figure('Position',[100,100,1200,500]);

% 9a. 3D surface + solver path
subplot(1,2,1);
surf(X, Y, Z, 'EdgeColor','none', 'FaceAlpha',0.8);
colormap(parula);
hold on;
% solver trajectory
hx = history(:,1);
hy = history(:,2);
hz = (1 - hx).^2 + 100*(hy - hx.^2).^2;
plot3(hx, hy, hz, 'r.-', 'MarkerSize',5, 'DisplayName','Solver Path');
plot3(hx(1), hy(1), hz(1), 'go', 'MarkerSize',8, 'DisplayName','Start');
plot3(x_opt(1), x_opt(2), f_opt, 'bo', 'MarkerSize',8, 'DisplayName','Minimum');
hold off;
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
title('Rosenbrock Function Surface Plot');
legend('Location','best');

% 9b. 2D contour + solver path
subplot(1,2,2);
levels = logspace(0, 3.5, 15);
contour(X, Y, Z, levels);
hold on;
plot(hx, hy, 'r.-', 'MarkerSize',5, 'DisplayName','Solver Path');
plot(hx(1), hy(1), 'go', 'MarkerSize',8, 'DisplayName','Start');
plot(x_opt(1), x_opt(2), 'bo', 'MarkerSize',8, 'DisplayName','Minimum');
hold off;
xlabel('x'); ylabel('y');
title('Rosenbrock Function Contour Plot');
legend('Location','best');

%% --- Callback function to record each iterate ---
function stop = outfun(x, optimValues, state)
    % Called by fminunc at each iteration.
    % Appends the current x to the global hist matrix.
    global hist;
    stop = false;  % don't halt optimization
    switch state
        case 'init'
            hist = [];  % clear at start
        case 'iter'
            hist(:, end+1) = x;
        case {'done', 'interrupt'}
            % nothing extra needed
    end
end
