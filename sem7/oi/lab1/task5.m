% Parameters
n = 1000; % Number of intervals for x
m = 1000; % Number of intervals for xi
c = 5;
a = 1; % Lower limit of integration for x
b = c; % Upper limit of integration for x
p = 1; % Lower limit for xi
q = 3; % Upper limit for xi
beta = 1/10; % Parameter beta
alpha = 1;

% Function f(x) = exp(i * beta * x)
f = @(x) exp(1i * beta * x);

K = @(xi, x) x ^ (alpha * xi - 1);

% Divide the interval [a, b] into n subintervals
hx = (b - a) / n;
x = a:hx:b;

% Define the ranges for xi
xi_ranges = {[1, 3], [0, 4], [0, 2], [2, 4], [1, 5], [3, 5]};

% Create a figure with 6 subplots
figure;

% Loop through the ranges and plot the amplitude and phase for each range
for i = 1:length(xi_ranges)
    % Extract the current range for xi
    p = xi_ranges{i}(1);
    q = xi_ranges{i}(2);

    % Divide the interval [p, q] into m subintervals
    hxi = (q - p) / m;
    xi = p:hxi:q;

    % Calculate the matrix A
    A = zeros(n+1, m+1);
    for j = 0:m
        for k = 0:n
            A(j+1, k+1) = K(xi(j+1), x(k+1));
        end
    end

    % Calculate the transformation F(l)
    F = A * f(x)' * hx;

    % Calculate the amplitude and phase of F(xi)
    amplitude = abs(F);
    phase = angle(F);

    % Subplot for amplitude
    subplot(6, 2, 2*i-1);
    plot(xi, amplitude);
    title(['Amplitude of F(ξ) for p = ', num2str(p), ', q = ', num2str(q)]);
    xlabel('ξ');
    ylabel('|F(ξ)|');
    grid on;

    % Subplot for phase
    subplot(6, 2, 2*i);
    plot(xi, phase);
    title(['Phase of F(ξ) for p = ', num2str(p), ', q = ', num2str(q)]);
    xlabel('ξ');
    ylabel('∠F(ξ)');
    grid on;
end
