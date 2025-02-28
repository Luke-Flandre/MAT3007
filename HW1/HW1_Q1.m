f = [-7.8 -7.1];
A = [1/8 1/4;1/2 1/6];
b = [90 80];
Aeq = []; Beq = []; lb = [0 0]; ub = [];
[X,Z] = linprog(f, A, b, Aeq, Beq, lb, ub);
disp(X)
disp(Z)