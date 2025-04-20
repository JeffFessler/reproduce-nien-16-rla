  function [xs, info] = pwls_hi_mem_admm(x, A, yi, R, varargin)
%|function [xs, info] = pwls_hi_mem_admm(x, A, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| using high-memory admm.
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|    x        [np 1]         initial estimate
%|    A        [nd np]        Gcone object, aij >= 0 required!
%|                            or sparse matrix (implies nsubset=1)
%|    yi       [nb na]        measurements (noisy sinogram data)
%|    R                       penalty object (see Reg1.m)
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    wi       [nb na]        weighting sinogram (default: [] for uniform)
%|    rho                     penalty parameter for sinogram split (default: 1)
%|    eta                     penalty parameter for roughness split (default: 1)
%|    npiter                  # of pcg iterations (default: 1)
%|    precon                  pcg preconditoner (default:1)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|
%| out
%|    xs       [np niter]     iterates
%|    info     [niter 1]      time
%|

if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.isave = [];
arg.wi = [];
arg.rho = 1;
arg.eta = 1;
arg.npiter = 1;
arg.precon = 1;
arg.userfun = @userfun_default;
arg.userarg = {};
arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);

cpu etic

wi = arg.wi;
if isempty(wi)
    wi = ones(size(yi));
end

% check input sinogram sizes
if ndims(yi) ~= 2
    fail 'bad yi size'
end
if ndims(wi) ~= 2
    fail 'bad wi size'
end

[nb na] = size(yi);

x = x(:);
np = length(x);
xs = zeros(np, length(arg.isave));
if any(arg.isave == 0)
    xs(:, arg.isave == 0) = x;
end
wi = wi(:);
yi = yi(:);
rho = single(arg.rho);
eta = single(arg.eta);

%info = zeros(niter,?); % do not initialize since size may change

% initilization
C = R.C1;
pot = R.pot{1};
thd = cell(R.M,1);
for ir = 1:R.M
    thd{ir} = R.wt.col(ir)/eta;
end
thd = cell2mat(thd);
G = rho*(A'*A)+eta*(C'*C);
b = wi.*yi;
% prox = @(z,thd) pot.meth.shrink(pot,z,thd);
prox = @(z,thd) huber_shrink(z,thd,pot.delta);

% den = rho*A'*(A*ones(np,1,'single'))+eta*(4*R.M);

Ax = A*x;
u = (b+rho*Ax)./(wi+rho);
d = -Ax+u;

Cx = C*x;
v = prox(Cx,thd);
e = -Cx+v;

% iterate
for iter = 1:arg.niter
    ticker(mfilename,iter,arg.niter);

    f = rho*A'*(u+d)+eta*C'*(v+e);
    x = qpwls_pcg2_no_warning(x,G,f,0,'precon',arg.precon,'niter',arg.npiter,'isave','last');
    
    % z = x; xold = x; told = 1;
    % for iiter = 1:arg.npiter
    %     num = rho*A'*(A*z-(u+d))+eta*C'*(C*z-(v+e));
    %     x = z-num./den;
    %     if (z-x)'*(x-xold)>0
    %         t = 1;
    %         z = x;
    %     else
    %         t = (1+sqrt(1+4*told^2))/2;
    %         z = x+(t-1)/told*(x-xold);
    %     end
    %     xold = x; told = t;
    % end
    
    Ax = A*x;
    u = (b+rho*(Ax-d))./(wi+rho);
    d = d-Ax+u;
    
    Cx = C*x;
    v = prox(Cx-e,thd);
    e = e-Cx+v;

    if any(arg.isave==iter)
        xs(:,arg.isave==iter) = x;
    end
    info(iter,:) = arg.userfun(x,arg.userarg{:});
end


% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');

% huber_shrink()
function out = huber_shrink(z, reg, delta)
out = z ./ (1 + reg);
big = delta * (1 + reg) < abs(z);
out(big) = z(big) .* (1 - reg(big) .* delta ./ abs(z(big)));
