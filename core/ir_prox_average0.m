  function [x, info] = ir_prox_average0(x, A, y, R, varargin)
%|function [x, info] = ir_prox_average0(x, A, y, R, [options])
%|
%| edge-preserving image restoration that finds a saddle-point of the
%| delta-approximate saddle-point function:
%| 
%| f(x,z;delta) = z'Kx - h*(z) + g(x) + delta/2 x'x ,
%|
%| where K = A, h(z) = 1/2 (z-y)'(z-y), and g(x) = R(x)+box(x), using an
%| accelerated first-order primal-dual algorithm with memory-efficient
%| proximal average updates.
%|
%| in
%|    x        [nx ny]        initial estimation
%|    A        [np np]        fatrix object for image blurring
%|    y        [nx ny]        noisy blury image
%|    R                       penalty object
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default: [0 inf])
%|    laa                     max eigenvalue of A'A
%|    tau0                    initial value of tau (default: 1/laa)
%|    delta                   strongly convex parameter (default: 1)
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
arg.laa = [];
arg.tau0 = [];
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

% maximum eigenvalue of A'A
L = arg.laa;
if isempty(L)
    L = max(A'*(A*one));
end

% initial primal step size
t = arg.tau0;
if isempty(t)
    t = 1/L;
end
s = 1/L/t;

% strongly convex parameter
d = arg.delta;
if d<0
    fail 'bad strongly convex parameter';
end

% initilization
x = max(min(tov(x),pixmax),pixmin);
xb = x;
z = A*x;
xold = x;
y = tov(y);
info = zeros(niter,1);

prox = @(u,a) (a*y+u)/(a+1);  % phi(z) = 1/2 (z-y)'(z-y)
[count,back] = loop_count_str(niter);

% image reconstruction
fprintf(['image recon: primal-dual with prox-average ' count],0);
for iter = 1:niter
    % z-update
    zp = z+s*A*xb;
    z = zp-s*prox(zp/s,1/s);
    % x-update
    tp = 1/(1/t+d);
    den = one/tp;
    xp = x-tp*(A'*z+d*x);
    x = edge_shrink(xp,R,den,'voxmax',box);
    % parameter update
    theta = 1/sqrt(1+d*t);
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
x = tom(x);

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
