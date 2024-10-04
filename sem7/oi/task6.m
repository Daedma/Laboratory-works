% Parameters
n = 1000; % Number of intervals for x
m = 1000; % Number of intervals for xi
c = 5;
a = 1; % Lower limit of integration for x
b = c; % Upper limit of integration for x
p = 1; % Lower limit for xi
q = 3; % Upper limit for xi
beta = 1/10; % Parameter beta
alpha_values = [0.1, 1, 4, 10, 15]; % Different values of alpha to investigate

% Function f(x) = exp(i * beta * x)
f = @(x) exp(1i * beta * x);

% Divide the interval [a, b] into n subintervals
hx = (b - a) / n;
x = a:hx:b;

% Divide the interval [p, q] into m subintervals
hxi = (q - p) / m;
xi = p:hxi:q;

% Create a figure with subplots for each alpha value
figure;

% Loop through the alpha values and plot the amplitude and phase for each value
for k = 1:length(alpha_values)
    alpha = alpha_values(k);
    K = @(xi, x) x ^ (alpha * xi - 1);

    % Calculate the matrix A
    A = zeros(n+1, m+1);
    for i = 0:m
        for j = 0:n
            A(i+1, j+1) = K(xi(i+1), x(j+1));
        end
    end

    % Calculate the transformation F(l)
    F = A * f(x)' * hx;

    % Calculate the amplitude and phase of F(xi)
    amplitude = abs(F);
    phase = angle(F);

    % Subplot for amplitude
    subplot(length(alpha_values), 2, 2*k-1);
    plot(xi, amplitude);
    title(['Amplitude of F(ξ) for α = ', num2str(alpha)]);
    xlabel('ξ');
    ylabel('|F(ξ)|');
    grid on;

    % Subplot for phase
    subplot(length(alpha_values), 2, 2*k);
    plot(xi, phase);
    title(['Phase of F(ξ) for α = ', num2str(alpha)]);
    xlabel('ξ');
    ylabel('∠F(ξ)');
    grid on;
end
