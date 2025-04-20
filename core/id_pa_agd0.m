  function [x, info] = id_pa_agd0(x, y, R, varargin)
%|function [x, info] = id_pa_agd0(x, y, R, [options])
%|
%| image denoise using the accelerated gradient descent method (zhong:14:asg)
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
mu = 0;
alp = 1;
% ell = L+b/alp;
ell = L+b*(0+1)^1.5;
eta = 1/ell;
info = nan(niter,1);

out = [];

[count,back] = loop_count_str(niter);

% image reconstruction
fprintf(['image recon: fast gradient descent with proximal average ' count],0);
for iter = 1:niter
    % w-update
    w = ((1-alp)*(mu+ell*alp)*x+ell*alp^2*z)/(mu*(1-alp)+ell*alp);
    % x-update
    xp = w-eta*(w-y);
    x = tom(edge_shrink(tov(xp),R,one/eta,'voxmax',box));
    % z-update
    z = z-(ell*(w-x)+mu*(z-w))/(ell*alp+mu);
    % update parameters
    alp = 2/(iter+2);
    % ell = L+b/alp;
    ell = L+b*(iter+1)^1.5;
    eta = 1/ell;
    
    out = [out; norm(w(:)-x(:)) 1/alp norm(w(:)-x(:))/alp];
    
    % update info
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf([back count],iter);
end
fprintf('\n');

figure; semilogy(out); legend('norm(w-x)','1/\alpha','norm(w-x)/\alpha');

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
