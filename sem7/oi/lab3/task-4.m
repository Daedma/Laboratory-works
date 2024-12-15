m = 5  
R = 5
n = 4096

function result = f(r, phi)
	R_1_5 = 10*r.^5 - 12*r.^3 + 3*r;
	result = R_1_5 .* exp(1i * 5 * phi);
  end

  
[X, Y] = meshgrid(linspace(-R, R, n), linspace(-R, R, n));

tic

f_values = f(sqrt(X.^2 + Y.^2), atan2(Y, X));

F = fft2(f_values);
F *= 1.5 * R/n * R/n;
F = fftshift(F);

toc

% Вычисление модуля и фазы
magnitude = abs(F);
phase = angle(F);

% Построение графиков
figure;
subplot(1, 2, 1);
imagesc(magnitude);
title('Модуль');
colorbar;

subplot(1, 2, 2);
imagesc(phase);
title('Фаза');
colorbar;