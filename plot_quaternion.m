%% read Odometry
clc, clear all, close all

load('trajectories_2.mat')

quat = x(7:10, :);

time = 1:size(quat, 2);

figure;
plot(time, quat(1, :), 'LineWidth', 1.5); hold on;
plot(time, quat(2, :), 'LineWidth', 1.5);
plot(time, quat(3, :), 'LineWidth', 1.5);
plot(time, quat(4, :), 'LineWidth', 1.5);
legend('w', 'x', 'y', 'z');
xlabel('Time index');
ylabel('Quaternion components');
title('Quaternion Components Over Time');
grid on;