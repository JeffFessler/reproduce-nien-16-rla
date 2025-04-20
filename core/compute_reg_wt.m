function wt = compute_reg_wt(R)

% wt = [];
% for mm = 1:R.M
%     wt = [wt; R.wt.col(mm)];
% end

nn = numel(R.mask);
wt = zeros(R.M*nn,1);
for mm = 1:R.M
    wt(1+(mm-1)*nn:mm*nn) = R.wt.col(mm);
end