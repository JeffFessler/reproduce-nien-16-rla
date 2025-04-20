  function [x, info] = ir_prox_average(x, A, y, R, varargin)
%|function [x, info] = ir_prox_average(x, A, y, R, [options])
%|
%| solve a smoothed edge-preserving image restoration problem:
%| 
%| minimize mu/2 (y-Ax)'(y-Ax) + delta/2 (Cx)'(Cx) + mu R(x)
%|    x
%|
%| using an accelerated first-order primal-dual algorithm:
%|
%| K = I
%| h(x) = mu/2 (y-Ax)'(y-Ax) + delta/2 (Cx)'(Cx) - gamma/2 x'x
%| g(x) = mu R(x) + gamma/2 x'x
%|
%| with memory-efficient proximal average updates.
%|
%| in
%|    x        [nx ny]        initial estimation
%|    A        [np np]        image blur object (see Gblur.m)
%|    y        [nx ny]        noisy blury image
%|    R                       penalty object (see Reg1.m)
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default: [0 inf])
%|    tau0                    initial value of tau (default: 1)
%|    mu                      cost function scaling parameter (default: 1)
%|    delta                   smooth parameter (default: 1)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat                    
%|
%| out
%|    x        [nx ny]        reconstructed image
%|    info     [niter 1]      output info
%|

if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.pixmax = inf;
arg.tau0 = 1;
arg.mu = 1;
arg.delta = 1;
arg.userfun = @userfun_default;
arg.userarg = {};
arg.chat = false;
arg = vararg_pair(arg, varargin);

% parameters
tov = @(x) x(R.mask);
tom = @(x) embed(x,R.mask);
np = sum(R.mask(:));
one = ones(np,1,'single');
niter = arg.niter;
cpu etic;

% box constraint
if length(arg.pixmax)==2
    pixmin = arg.pixmax(1);
    pixmax = arg.pixmax(2);
elseif length(arg.pixmax)==1
    pixmin = 0;
    pixmax = arg.pixmax;
else
    fail 'bad pixmax'
end
box = [pixmin pixmax];

% initial primal step size
t = arg.tau0;
s = 1/t;  % K = I

% filter spectrum
faa = A.psf_fft.*conj(A.psf_fft);
fcc = 0;
ej = zeros(size(x)); ej(2,2) = 1;
for i = 1:R.M
    ci = R.C1s{i}*ej; ci = ci(1:3,1:3);
    Ci = Gblur(R.mask,'psf',ci,'type','fft,same');
    bi = R.beta(i)/max(R.beta);
    fcc = fcc+bi*(Ci.psf_fft.*conj(Ci.psf_fft));
end

% scaling parameter
m = arg.mu;
if m<=0
    fail 'bad scaling parameter';
end

% smooth parameter
d = arg.delta;
if isempty(d)
    d = m*(np-sum(faa(:)))/sum(fcc(:));  % gamma ~ mu
end
fprintf(['choose d = ' num2str(d) '... ']);
g = min(m*faa(:)+d*fcc(:));
if g<0
    fail 'the cost function is not strongly convex';
else
    fprintf([num2str(g) '-strongly convex cost function\n']);
end
fss = m*faa+d*fcc-g;

% initilization
x = max(min(x,pixmax),pixmin);
xb = x;
z = x;
xold = x;
Ay = A'*y;
info = zeros(niter,1);

[count,back] = loop_count_str(niter);

% image reconstruction
fprintf(['image recon: primal-dual with prox-average ' count],0);
for iter = 1:niter
    % z-update
    zp = z+s*xb;
    z = zp-s*ifftn(fftn(m*Ay+zp)./(fss+s));
    % x-update
    tp = 1/(1/t+g);
    den = one/tp/m;
    xp = x-tp*(z+g*x);
    % x = tom(edge_shrink(tov(xp),R,den,'voxmax',box));
    x = tom(edge_shrink_double(double(tov(xp)),R,double(den),'voxmax',box));
    % parameter update
    theta = 1/sqrt(1+g*t);
    t = t*theta;
    s = s/theta;
    % xb-update
    xb = x+theta*(x-xold);
    xold = x;
    
    % update info
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf([back count],iter);
end
fprintf('\n');

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
