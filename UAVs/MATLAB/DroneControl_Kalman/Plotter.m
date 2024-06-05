close all
% plot simout results
x = -out.Data(out.time<70, 1);
y = -out.Data(out.time<70, 2);
z = -out.Data(out.time<70, 3);
x_ref = reference.Data(reference.time<70, 1);
y_ref = reference.Data(reference.time<70, 2);
z_ref = reference.Data(reference.time<70, 3);

% 3D Movement Plot vs Desired
figure
plot3(x, y, z, 'color', 'g')
hold on
plot3(x_ref, y_ref, z_ref, 'r--')
title('3D plot of quadrotor position')
legend("Drone Position", "Drone Desired Position")
xlabel('x')
ylabel('y')
zlabel('z')
title("Desired vs Actual Dron Position")

% x, xdot, y, ydot, z, zdot, phi, phidot, theta, thetadot, psi, psidot
states = state_out.Data(state_out.time<70, :);
state_names = ["x", "xdot", "y", "ydot", "z", "zdot", "phi", "phidot", "theta", "thetadot", "psi", "psidot"];
% Plot for all states in a 4x3 grid
figure
for i = 1:12
    subplot(4, 3, i)
    plot(state_out.time(state_out.time<70), states(:, i))
    title("State " + state_names(i))
end

% x_control, y_control, z_control, psi_control
controls = control_out.Data(control_out.time<70, :);
control_names = ["x_control", "y_control", "z_control", "psi_control"];
% Plot for all controls in a 2x2 grid
figure
for i = 1:4
    subplot(2, 2, i)
    plot(control_out.time(control_out.time<70), controls(:, i))
    title("Control " + control_names(i))
end
