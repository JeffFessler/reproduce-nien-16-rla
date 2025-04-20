  function [x, info] = ir_fgm1(x, A, y, R, varargin)
%|function [x, info] = ir_fgm1(x, A, y, R, [options])
%|
%| solve a edge-preserving image restoration problem:
%| 
%| minimize 1/2 (y-Ax)'(y-Ax) + R(x)
%|    x
%|
%| using the fast gradient method.
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
%|    reg_type                type of regularizer
%|                            options: 'max' (default), 'huber', and 'prox'
%|    restart                 adaptive restart (default: 1)
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
arg.reg_type = 'max';
arg.restart = 1;
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
restart = arg.restart;
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
proj = @(x) min(max(x,pixmin),pixmax);

% spectrum radius of A
faa = A.psf_fft.*conj(A.psf_fft);
Dden = max(faa(:));

% regularization type
if strcmp(arg.reg_type,'prox')
    is_prox = 1;
elseif strcmp(arg.reg_type,'max') || strcmp(arg.reg_type,'huber')
    is_prox = false;
    is_huber = strcmp(arg.reg_type,'huber');
    Rden = R.denom(R,zeros(R.np,1,'single'));
else
    fail 'bad reg_type'
end

% initilization
x = proj(tov(x));
y = tov(y);
z = x;
xold = x;
told = 1;
info = zeros(niter,1);

[count,back] = loop_count_str(niter);

% image reconstruction
fprintf(['image recon: fast gradient method ' count],0);
for iter = 1:niter
    Dnum = A'*(A*z-y);
    if is_prox
        num = Dnum;
        den = Dden;
        x = edge_shrink(z-num./den,R,den,'voxmax',[pixmin pixmax]);
    else
        num = Dnum+R.cgrad(R,z);
        if is_huber
            den = Dden+R.denom(R,z);
        else
            den = Dden+Rden;
        end
        x = proj(z-num./den);
    end
    
    if restart && (z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
    end
    xold = x;
    told = t;
    
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
