function rho = rho_sequence(k,lb)
% rho = rho_sequence(k,lb) returns the decreasing rho sequence (with a
% lower bound lb) used for the deterministic downward continuation in the
% OS-LALM algorithm.
% 
% Copyright 2013-11-19, Hung Nien, University of Michigan

if sum(k<0)>0
    error 'The index of rho sequence should be a positive integer!';
end

if nargin<2
    lb = 0;
end

% the designed decreasing sequence in the paper
invc = (pi/2./k).^2;
rho = max(2*sqrt(invc.*(1-invc)).*(k>1)+(k==1),lb);

% the heuristic O(1/k) decreasing sequence
% rho = max(4./(k+4),lb);