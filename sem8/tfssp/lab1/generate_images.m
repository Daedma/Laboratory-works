1;

function [image, x, y] = generate_image(n, phi0, omega, scaling, size_img)
	x = linspace(-scaling, scaling, size_img);
	y = linspace(-scaling, scaling, size_img);
	[X, Y] = meshgrid(x, y);

	% Радиус
	r = sqrt(X.^2 + Y.^2);

	% Азимутальный угол
	phi = atan2(Y, X);

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

	% Создание суммарного изображения
	image = zeros(size_img * 2, size_img * 2);
	image(1:2:end, 1:2:end) = images(:, :, 1);
	image(2:2:end, 2:2:end) = images(:, :, 2);
	image(2:2:end, 1:2:end) = images(:, :, 3);
	image(1:2:end, 2:2:end) = images(:, :, 4);
end

function image = add_noise(image, noise_level)
	image = image + noise_level * randn(size(image));
end

function image = apply_threshold(image, threshold)
	max_intensity = max(image(:));
	image(image > threshold * max_intensity) = threshold * max_intensity;
end

function plot_image(image, x, y)
	imagesc(x, y, image);
	colormap(jet);
	colorbar;
	title('Суммарное поляризационное изображение');
	xlabel('x, мкм');
	ylabel('y, мкм');
end

function save_image(image, filename)
	imwrite(image, filename);
end

% Параметры для перебора
n_values = [16]; % Порядок векторной моды
phi0_values = [pi/6]; % Начальный угол поворота моды
omega_values = [8]; % Ширина моды
noise_level_values = [0.025]; % Уровень шума (сигнал/шум)
threshold_values = [0.92]; % Порог зашкаливания

size_img = 300; % Размер изображения
scale = 15; % Масштаб изображения

% Цикл по всем комбинациям параметров
for n = n_values
    for phi0 = phi0_values
        for omega = omega_values
            for noise_level = noise_level_values
                for threshold = threshold_values
                    [image, x, y] = generate_image(n, phi0, omega, scale, size_img);
                    image = add_noise(image, noise_level);
                    image = apply_threshold(image, threshold);
                    filename = sprintf('n%d_phi0%d_omega%d_noise%d_threshold%d.png', floor(n*1000), floor(phi0*1000), floor(omega*1000), floor(noise_level*1000), floor(threshold*1000));
					save_image(image, filename)
                end
            end
        end
    end
end