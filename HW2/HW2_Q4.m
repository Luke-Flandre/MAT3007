cvx_begin
variables xa xb xc xd xe xf xg xh xi xj
minimize xa + xb + xc + xd + xe + xf + xg + xh + xi + xj
subject to
    xa + xb >= 1
    xa + xe >= 1
    xa + xf >= 1
    xb + xc >= 1
    xb + xg >= 1
    xc + xh >= 1
    xc + xd >= 1
    xd + xi >= 1
    xd + xe >= 1
    xe + xj >= 1
    xf + xi >= 1
    xf + xh >= 1
    xg + xi >= 1
    xg + xj >= 1
    xh + xj >= 1
    0 <= xa, xb, xc, xd, xe, xf, xg, xh, xi, xj <= 1
cvx_end
