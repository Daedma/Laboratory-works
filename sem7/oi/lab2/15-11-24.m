
% Параметры
N = 256; % Размерность вектора f
M = 1024; % Размерность после дополнения нулями
a = 5;
hx = 2 * a / N; % Шаг по x
sigma = 1/sqrt(2*pi);
H = @(x) 32*x.^5 - 160*x.^3 +120*x;
input_field = @(x) H(x/sigma) .* exp(-(x/sigma).^2 / 2);

% Шаг 1: Дискретизация входной функции f(x)
x = linspace(-a, a, N);
f = input_field(x);

% Шаг 2: Дополнение вектора f нулями до размерности M
f_padded = [zeros(1, (M-N)/2), f, zeros(1, (M-N)/2)];

% Шаг 3: Разбить вектор f на две половины и поменять их местами
f_swapped = [f_padded((M/2+1):end), f_padded(1:(M/2))];

% Шаг 4: Выполнить БПФ от f и умножить результат на шаг hx
F1 = fft(f_swapped) * hx;

% Шаг 5: Разбить вектор F на две половины и поменять их местами
F1 = [F1((M/2+1):end), F1(1:(M/2))];

% Шаг 6: «Вырезать» центральную часть вектора F, оставив центральные N элементов
F1 = F1((M/2-N/2+1):(M/2+N/2));

% Шаг 7: Пересчитать область задания функции Fa(u) по формуле (7)
b = N^2 / (4 * a * M);
u = linspace(-b, b, N);

% Финитное преобразование Фурье методом прямоугольников
F2 = zeros(size(u));

for k = 1:length(u)
    integrand = f .* exp(-1i * 2 * pi * u(k) * x);
    F2(k) = sum(integrand) * hx;
end

% Построение графика амплитуды
figure;
subplot(2, 1, 1);
plot(x, abs(f));
title('Амплитуда моды Гаусса-Эрмита');
xlabel('x');
ylabel('Амплитуда');
grid on;

% Построение графика фазы
subplot(2, 1, 2);
plot(x, angle(f));
title('Фаза моды Гаусса-Эрмита');
xlabel('x');
ylabel('Фаза');
grid on;

% Построение графика амплитуды
figure;
subplot(2, 1, 1);
plot(u, abs(F1));
hold on;
plot(u, abs(F2));
plot(x, abs(f));
hold off;
title('Амплитуда F');
xlabel('x');
ylabel('Амплитуда');
legend('fft', 'rect', 'Входное поле');
grid on;

% Построение графика фазы
subplot(2, 1, 2);
plot(u, angle(F1));
hold on;
plot(u, angle(F2));
plot(x, angle(f));
hold off;
title('Фаза F');
xlabel('x');
ylabel('Фаза');
legend('fft', 'rect', 'Входное поле');
grid on;
