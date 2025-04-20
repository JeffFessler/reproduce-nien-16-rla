  function [x, info] = id_pa_apg0(x, y, R, varargin)
%|function [x, info] = id_pa_apg0(x, y, R, [options])
%|
%| image denoise using the accelerated proximal gradient method (yu:13:baa)
%| with memory-efficient proximal average updates.
%|
%| in
%|    x        [nx ny]        initial estimation
%|    y        [nx ny]        noisy blury image
%|    R                       penalty object (see Reg1.m)
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default: [0 inf])
%|    b                       step size parameter (default: 1)
%|    etamin                  minimum eta (default: -inf)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat                    
%|
%| out
%|    x        [nx ny]        reconstructed image
%|    info     [niter 1]      output info
%|

if nargin < 3, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.pixmax = inf;
arg.b = 1;
arg.etamin = -inf;
arg.userfun = @userfun_default;
arg.userarg = {};
arg.chat = false;
arg = vararg_pair(arg, varargin);

% parameters
tov = @(x) x(R.mask);
tom = @(x) embed(x,R.mask);
np = sum(R.mask(:));
one = ones(np,1,'single');
L = 1; % lipschitz constant
niter = arg.niter;
b = arg.b;
etamin = arg.etamin;
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

% initilization
x = max(min(x,pixmax),pixmin);
z = x;
told = 1;
xold = x;
ell = L+b*0;
eta = 1/ell;
info = zeros(niter,1);
param = eta;

[count,back] = loop_count_str(niter);

% image reconstruction
fprintf(['image recon: fast gradient descent with proximal average ' count],0);
for iter = 1:niter
    % x-update
    zp = z-eta*(z-y);
    x = tom(edge_shrink(tov(zp),R,one/eta,'voxmax',box));
    % z-update
    t = (1+sqrt(1+4*told^2))/2;
    z = x+(told-1)/t*(x-xold);
    % update parameters
    ell = L+b*iter;
    eta = max(1/ell,etamin);
    told = t;
    xold = x;
    param = [param; eta];
    
    % update info
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf([back count],iter);
end
fprintf('\n');
figure; semilogy(param);

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
