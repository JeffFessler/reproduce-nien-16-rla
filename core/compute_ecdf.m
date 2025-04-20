function [Fx,x] = compute_ecdf(y,nBin)

x = linspace(min(y),max(y),nBin+1);

n = histc(y,x);
Fx = cumsum(n)/sum(n);
