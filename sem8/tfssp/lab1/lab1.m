% Параметры
n = 1; % Порядок векторной моды
phi0 = 0; % Начальный угол поворота моды
omega = 6; % Ширина моды
noise_level = 0.0; % Уровень шума (сигнал/шум)
threshold = 1.0; % Порог зашкаливания

% Размер изображения
size_img = 300;
[x, y] = meshgrid(linspace(-omega, omega, size_img), linspace(-omega, omega, size_img));

% Радиус
r = sqrt(x.^2 + y.^2);

% Азимутальный угол
phi = atan2(y, x);

% Компоненты напряженности электрического поля
Ex = r .* exp(-r.^2 / omega^2) .* cos(n * (phi - phi0));
Ey = r .* exp(-r.^2 / omega^2) .* sin(n * (phi - phi0));

% Поляризационные изображения
images = zeros(size_img, size_img, 4);

% Углы поляризатора
theta_values = [0, pi/2, pi/4, -pi/4];

for i = 1:4
    theta = theta_values(i);
    if theta == 0
        images(:, :, i) = Ex.^2;
    elseif theta == pi/2
        images(:, :, i) = Ey.^2;
    else
		Ex_out = Ex .* cos(theta)^2 - Ey .* cos(theta) * sin(theta);
		Ey_out = -Ex .* cos(theta) * sin(theta) + Ey .* sin(theta)^2;
        images(:, :, i) = Ex_out.^2 + Ey_out.^2;
    end
end

% Добавление шума
images = images + noise_level * randn(size(images));

% Зашкаливание
max_intensity = max(images(:));
images(images > threshold * max_intensity) = threshold * max_intensity;

% Создание суммарного изображения
final_image = zeros(size_img * 2, size_img * 2);
final_image(1:2:end, 1:2:end) = images(:, :, 1);
final_image(2:2:end, 2:2:end) = images(:, :, 2);
final_image(2:2:end, 1:2:end) = images(:, :, 3);
final_image(1:2:end, 2:2:end) = images(:, :, 4);

% Отображение суммарного изображения
imagesc(linspace(-omega, omega, size_img * 2), linspace(-omega, omega, size_img * 2), final_image);
colormap(jet);
colorbar;
title('Суммарное поляризационное изображение');

% Добавление подписей осей
xlabel('x, мкм');
ylabel('y, мкм');