% Для аналитики применить не финитное
% преобразование. Использовать свойства
% преобразования Фурье.
% Параметры
N = 500; % Размерность вектора f
M = 2048; % Размерность после дополнения нулями
a = 5;
hx = 2 * a / N; % Шаг по x
input_field = @(x, y) exp(-x.^2 - y.^2);

% Шаг 1: Дискретизация входной функции f(x)
x = linspace(-a, a, N);
y = linspace(-a, a, N);
[X, Y] = meshgrid(x, y);
f = input_field(X, Y);


% Вывод результатов
figure;
subplot(2,1,1);
imagesc(x, y, abs(f));
title('Амплитуда f(x, y)');
xlabel('x');
ylabel('y');
colorbar;

subplot(2,1,2);
imagesc(x, y, angle(f));
title('Фаза f(x, y)');
xlabel('x');
ylabel('y');
colorbar;

