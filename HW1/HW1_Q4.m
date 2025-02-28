M = 100;
W = [M, 5, 4, M, M, M, M, M;
5, M, M, 3, M, 7, M, M;
4, M, M, M, 1, 2, M, M;
M, 3, M, M, 2, M, M, M;
M, M, 1, 2, M, M, 2, 5;
M, 7, 2, M, M, M, M, 3;
M, M, M, M, 2, M, M, 1;
M, M, M, M, 5, 3 ,1, M];
[m,n] = size(W);

cvx_begin
    variable x(n,n);
    minimize(sum(sum(W.*x)));
    subject to
        sum(x(1, :)) - sum(x(:, 1)) == 1;
        sum(x(:, n)) - sum(x(n, :)) == 1;
        for i = 2: n - 1
            sum(x(i,:)) - sum(x(:,i)) == 0;
        end
        max(max(x)) <= 1;
        min(min(x)) >= 0;
cvx_end
disp(x)
