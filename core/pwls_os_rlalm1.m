  function [xs, info] = pwls_os_rlalm1(x, Ab, yi, R, varargin)
%|function [xs, info] = pwls_os_rlalm1(x, Ab, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| using relaxed linearized augmented Lagrangian with (optionally relaxed)
%| ordered subsets.
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
%|    rho                     al penalty parameter (default 1)
%|    alpha                   relaxation parameter (default 1)
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
arg.rho = 1;
arg.alpha = 1;
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
denom0 = denom;
denom(denom==0) = inf; % trick: prevents pixels where denom=0 being updated

if strcmp(arg.reg_type,'pa')
    is_pa = true;
elseif strcmp(arg.reg_type,'max') || strcmp(arg.reg_type,'huber')
    is_pa = false;
    is_huber = strcmp(arg.reg_type,'huber');
    Rdenom = R.denom(R,zeros(R.np,1,'single'));
else
    fail 'bad reg_type'
end

rho = arg.rho;
% if isnumeric(arg.rho)
%     if arg.rho>0
%         rho = str2func(['@(k)' num2str(arg.rho)]);
%     else
%         fail 'illegal al penalty parameter rho?!';
%     end
% else
%     rho = str2func(arg.rho);
% end

alpha = arg.alpha;

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
ib = starts(nblock);
ia = ib:nblock:na;
grad = nblock*Ab{ib}'*(col(wi(:,ia)).*(Ab{ib}*x-col(yi(:,ia))));
g = rho(1)*grad;
h = denom0.*x-grad;

thd = norm(rho*(g+h))*1e-2;
res_norm = [];

% xold = x;
gg = g; hh = h;
relax = relax0;
limit = arg.limit0;
ep_old = inf;

% iterate
fprintf(['relaxed ordered-subset linearized al method (' arg.reg_type ')... ' count],0);
for iter = 1:arg.niter
    % ticker(mfilename,iter,arg.niter)

    % loop over subsets
    for iset = 1:nblock
        k = nblock*(iter-1)+iset;

        if is_pa
            gamma = (rho-1)*g+rho*h;
            den = rho*denom/relax;
            x = edge_shrink(gamma./den,R,den,'voxmax',[pixmin pixmax]);
        else
            num = rho*denom0.*x+(1-rho)*g-rho*h+R.cgrad(R,x);
            if is_huber
                den = rho*denom/relax+R.denom(R,x);
            else
                den = rho*denom/relax+Rdenom;
            end
            x = proj(x-num./den);
        end
        
        ib = starts(iset);
        ia = ib:nblock:na;
        grad = nblock*Ab{ib}'*(col(wi(:,ia)).*(Ab{ib}*x-col(yi(:,ia))));
        
        g = (rho*(alpha*grad+(1-alpha)*g)+g)/(rho+1);
        h = alpha*(denom0.*x-grad)+(1-alpha)*h;

        res = rho*((g-gg)+(h-hh));
        res_norm = [res_norm; norm(res)];
        if 0%norm(res)<thd
            rho = rho*2;
            thd = thd/2;
        end
        gg = g; hh = h;
        
%        ep = norm(x-xold)/norm(xold);
%        if ep<=limit || ep>10*ep_old
%            relax = relax0/(1+relax_rate*(nblock*(iter-1)+iset));
%            ep_old = ep;
%        end
%        xold = x;
    end
    
%     if mod(iter,arg.niter/5)==0
%         rho = rho*2;
%     end

    if any(arg.isave == iter)
        xs(:, arg.isave == iter) = x;
    end
    info(iter,:) = arg.userfun(x, arg.userarg{:});
    fprintf([back count],iter);
end
fprintf('\n');

figure; plot(res_norm);

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
