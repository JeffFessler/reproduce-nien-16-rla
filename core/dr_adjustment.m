function ut = dr_adjustment(u,t,epsilon)

% compute the empirical cdf
% [f,x] = ecdf(u);
% Fu = interp1(x(2:end),f(2:end),u);
% Fu(u>max(x)) = 1;
% Fu(u<min(x)) = 0;
[f,x] = compute_ecdf(u,ceil(length(u)/10));
Fu = interp1(x,f,u);

% gamma correction and thresholding
ut = max(Fu.^t,epsilon);