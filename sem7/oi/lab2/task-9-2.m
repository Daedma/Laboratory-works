% Для аналитики применить не финитное
% преобразование. Использовать свойства
% преобразования Фурье.
% Параметры
N = 500; % Размерность вектора f
M = 2048; % Размерность после дополнения нулями
a = 5;
hx = 2 * a / N; % Шаг по x
input_field = @(x, y) sin(4*pi*x) .* sin(4*pi*y);
analytical_F = @(u, v) (-a*1i*sinc(2*(2-u)*a) + a*1i*sinc(2*(2+u)*a)) .* (-a*1i*sinc(2*(2-v)*a) + a*1i*sinc(2*(2+v)*a));

% Шаг 7: Пересчитать область задания функции Fa(u) по формуле (7)
b = N^2 / (4 * a * M);
u = linspace(-b, b, N);
v = linspace(-b, b, N);
[U, V] = meshgrid(u, v);

F = analytical_F(U, V);

% Вывод результатов
figure;
subplot(2,1,1);
imagesc(u, v, abs(F));
title('Амплитуда F(u, v)');
xlabel('u');
ylabel('v');
colorbar;

subplot(2,1,2);
imagesc(u, v, angle(F));
title('Фаза F(u, v)');
xlabel('u');
ylabel('v');
colorbar;

