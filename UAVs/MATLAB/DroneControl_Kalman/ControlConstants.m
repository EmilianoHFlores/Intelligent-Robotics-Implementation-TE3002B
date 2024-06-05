clc
clear
close all
% quadrotor specifications
m = 0.0686; 
g = 9.81;
Ixx = 0.0686e-3;
Iyy = 0.092e-3;
Izz = 0.1366e-3;
kl = 0.0107;

A = [0 1 0 0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 g 0 0 0;
     0 0 0 1 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 -g 0 0 0 0 0;
     0 0 0 0 0 1 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 1 0 0 0 0;
     0 0 0 0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0 1 0 0;
     0 0 0 0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0 0 0 1;
     0 0 0 0 0 0 0 0 0 0 0 0];

B = [0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    0 0 0 0;
    1/m 0 0 0;
    0 0 0 0;
    0 1/Ixx 0 0;
    0 0 0 0;
    0 0 1/Iyy 0;
    0 0 0 0;
    0 0 0 1/Izz];

C = [1 0 0 0 0 0 0 0 0 0 0 0;
     0 0 1 0 0 0 0 0 0 0 0 0;
     0 0 0 0 1 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0 0 1 0];

% Controllability matrix
Cctr = ctrb(A,B);

% Rank of controllability matrix
rank(Cctr)

% Desired poles
% P = [-0.1 -0.1 -0.1 -0.5 -0.5 -0.5 -1 -1 -1 -1 -15 -15];
P = [-11 -11 -9 -9 -7 -7 -8 -4 -15 -15 -3 -3];
P = P.*1;

% Gain matrix
K = place(A,B,P);

% k bar
k_bar = inv(A - B*K);
k_bar = pinv(C*k_bar*B);
