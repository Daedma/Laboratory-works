% Параметры
n = 1000; % Количество интервалов разбиения для x
m = 1000; % Количество интервалов разбиения для xi
c = 5;
a = 1; % Нижний предел интегрирования для x
b = c; % Верхний предел интегрирования для x
p = 1; % Нижний предел для xi
q = 3; % Верхний предел для xi
beta = 1/10; % Параметр beta
alpha = 1;

% Функция f(x) = exp(i * beta * x)
f = @(x) exp(1i * beta * x);

K = @(xi, x) x ^ (alpha * xi - 1);

% Разбиение отрезка [a, b] на n интервалов
hx = (b - a) / n;
x = a:hx:b;

% Разбиение отрезка [p, q] на m интервалов
hxi = (q - p) / m;
xi = p:hxi:q;

% Вычисление матрицы A
A = zeros(n+1, m+1);
for i = 0:m
    for j = 0:n
        A(i+1, j+1) = K(xi(i+1), x(j+1));
    end
end

% Вычисление преобразования F(l)
F = A * f(x)' * hx;

% Calculate the amplitude and phase of F(xi)
amplitude = abs(F);
phase = angle(F);

% Plot the amplitude and phase in the same figure
figure;

% Subplot for amplitude
subplot(2,1,1);
plot(xi, amplitude);
title('Amplitude of F(ξ)');
xlabel('ξ');
ylabel('|F(ξ)|');
grid on;

% Subplot for phase
subplot(2,1,2);
plot(xi, phase);
title('Phase of F(ξ)');
xlabel('ξ');
ylabel('∠F(ξ)');
grid on;
