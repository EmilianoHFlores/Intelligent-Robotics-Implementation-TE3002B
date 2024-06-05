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
