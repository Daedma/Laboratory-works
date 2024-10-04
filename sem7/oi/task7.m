% Parameters
n = 1000; % Number of intervals for x
m = 1000; % Number of intervals for xi
a = 1; % Lower limit of integration for x
c_values = [3, 5, 7, 10]; % Different values of c to investigate
p = 1; % Lower limit for xi
q = 3; % Upper limit for xi
beta = 1/10; % Parameter beta
alpha = 1;

% Function f(x) = exp(i * beta * x)
f = @(x) exp(1i * beta * x);

K = @(xi, x) x ^ (alpha * xi - 1);

% Create a figure with subplots for each c value
figure;

% Loop through the c values and plot the amplitude and phase for each value
for k = 1:length(c_values)
    c = c_values(k);
    b = c; % Upper limit of integration for x

    % Divide the interval [a, b] into n subintervals
    hx = (b - a) / n;
    x = a:hx:b;

    % Divide the interval [p, q] into m subintervals
    hxi = (q - p) / m;
    xi = p:hxi:q;

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
    subplot(length(c_values), 2, 2*k-1);
    plot(xi, amplitude);
    title(['Amplitude of F(ξ) for c = ', num2str(c)]);
    xlabel('ξ');
    ylabel('|F(ξ)|');
    grid on;

    % Subplot for phase
    subplot(length(c_values), 2, 2*k);
    plot(xi, phase);
    title(['Phase of F(ξ) for c = ', num2str(c)]);
    xlabel('ξ');
    ylabel('∠F(ξ)');
    grid on;
end
