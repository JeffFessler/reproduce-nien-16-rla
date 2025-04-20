  function [xs, info] = pwls_os_fgm1(x, Ab, yi, R, varargin)
%|function [xs, info] = pwls_os_fgm1(x, Ab, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| using fast gradient method with (optionally relaxed) ordered subsets.
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|    x        [np 1]         initial estimate
%|    Ab       [nd np]        Gblock object, aij >= 0 required!
%|                            or sparse matrix (implies nsubset=1)
%|    yi       [nb na]        measurements (noisy sinogram data)
%|    R                       penalty object (see Reg1.m)
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    wi       [nb na]        weighting sinogram (default: [] for uniform)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default [0 inf])
%|    denom    [np 1]         precomputed denominator
%|    aai      [nb na]        precomputed row sums of |Ab|
%|    relax0   [1] or [2]     relax0 or (relax0, relax_rate)
%|    limit0                  limit of relative step size (default: 0, i.e., no relaxation)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat
%|    mom_type                type of momentum: 'std' (default) | 'opt'
%|    reg_type                type of regularizer: 'max' (default) | 'huber' | 'pa'
%|
%| out
%|    xs       [np niter]     iterates
%|    info     [niter 1]      time
%|

if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.isave = 'last';
arg.userfun = @userfun_default;
arg.userarg = {};
arg.pixmax = inf;
arg.chat = false;
arg.wi = [];
arg.aai = [];
arg.relax0 = 1;
arg.limit0 = 0;
arg.denom = [];
arg.mom_type = 'std';
arg.reg_type = 'max';
arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);

Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');
starts = subset_start(nblock);

cpu etic

wi = arg.wi;
if isempty(wi)
    wi = ones(size(yi));
end
if isempty(arg.aai)
    arg.aai = reshape(sum(Ab'),size(yi)); % a_i = sum_j |a_ij|
                    % requires real a_ij and a_ij >= 0
end

% check input sinogram sizes for OS
if (ndims(yi)~=2) || (size(yi,2)==1 && nblock>1)
    fail 'bad yi size'
end
if (ndims(wi)~=2) || (size(wi,2)==1 && nblock>1)
    fail 'bad wi size'
end

relax0 = arg.relax0(1);
if length(arg.relax0)==1
    relax_rate = 0;
elseif length(arg.relax0)==2
    relax_rate = arg.relax0(2);
else
    fail 'bad relax0'
end

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

% likelihood denom, if not provided
denom = arg.denom;
if isempty(denom)
    denom = Ab'*col(arg.aai.*wi); % requires real a_ij and a_ij >= 0
end
denom(denom==0) = inf; % trick: prevents pixels where denom=0 being updated

if strcmp(arg.mom_type,'std') || strcmp(arg.mom_type,'opt')
    is_opt = strcmp(arg.mom_type,'opt');
else
    fail 'bad mom_type'
end

if strcmp(arg.reg_type,'pa')
    is_pa = true;
elseif strcmp(arg.reg_type,'max') || strcmp(arg.reg_type,'huber')
    is_pa = false;
    is_huber = strcmp(arg.reg_type,'huber');
    Rdenom = R.denom(R,zeros(R.np,1,'single'));
else
    fail 'bad reg_type'
end

[~,na] = size(yi);

x = proj(x(:));
np = length(x);
xs = zeros(np,length(arg.isave));
if any(arg.isave==0)
    xs(:,arg.isave==0) = x;
end
[count,back] = loop_count_str(arg.niter);

%info = zeros(niter,?); % do not initialize since size may change

% initilization
z = x;
xold = x;
told = 1;
relax = relax0;
limit = arg.limit0;
ep_old = inf;

% iterate
fprintf(['fast gradient method (' arg.reg_type ')... ' count],0);
for iter = 1:arg.niter
    % ticker(mfilename,iter,arg.niter)

    % loop over subsets
    for iset = 1:nblock
        ib = starts(iset);
        ia = ib:nblock:na;
        num = nblock*Ab{ib}'*(col(wi(:,ia)).*(Ab{ib}*z-col(yi(:,ia))));
        if is_pa
            den = denom/relax;
            x = edge_shrink(z-num./den,R,den,'voxmax',[pixmin pixmax]);
        else
            num = num+R.cgrad(R,z);
            if is_huber
                den = denom+R.denom(R,z);
            else
                den = denom+Rdenom;
            end
            x = proj(z-relax*num./den);
        end
        t = (1+sqrt(1+4*told^2))/2;
        if is_opt
            z = x+(told-1)/t*(x-xold)+told/t*(x-z);
        else
            z = x+(told-1)/t*(x-xold);
        end
        ep = norm(x-xold)/norm(xold);
        if ep<=limit || ep>10*ep_old
            relax = relax0/(1+relax_rate*(nblock*(iter-1)+iset));
            ep_old = ep;
        end
        xold = x;
        told = t;
    end

    if any(arg.isave == iter)
        xs(:, arg.isave == iter) = x;
    end
    info(iter,:) = arg.userfun(x, arg.userarg{:});
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
