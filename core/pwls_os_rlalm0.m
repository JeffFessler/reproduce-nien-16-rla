  function [xs, info] = pwls_os_rlalm0(x, Ab, yi, R, varargin)
%|function [xs, info] = pwls_os_rlalm0(x, Ab, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| using naive relaxed linearized augmented lagrangian method with
%| (optionally relaxed) ordered subsets.
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|    x        [np 1]         initial estimate
%|    Ab       [nd np]        Gblock object, aij >= 0 required!
%|                            or sparse matrix (implies nsubset=1)
%|    yi       [nb na]        measurements (noisy sinogram data)
%|    R                       penalty object (see Reg1.m), can be []
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    nfiter                  # of fista iterations (default: 1)
%|    wi       [nb na]        weighting sinogram (default: [] for uniform)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default [0 inf])
%|    denom    [np 1]         precomputed denominator
%|    aai      [nb na]        precomputed row sums of |Ab|
%|    relax0   [1] or [2]     relax0 or (relax0, relax_rate)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat
%|    rho                     al penalty parameter (default 1)
%|    alpha                   relaxation parameter (default 1)
%|
%| out
%|    xs       [np niter]     iterates
%|    info     [niter 1]      time
%|

if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.nfiter = 1;
arg.isave = [];
arg.userfun = @userfun_default;
arg.userarg = {};
arg.pixmax = inf;
arg.chat = false;
arg.wi = [];
arg.aai = [];
arg.relax0 = 1;
arg.denom = [];
arg.scale_nblock = true; % traditional scaling
arg.update_even_if_denom_0 = true;
arg.rho = 1;
arg.alpha = 1;
arg = vararg_pair(arg,varargin);

arg.isave = iter_saver(arg.isave,arg.niter);

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
    error relax
end

if length(arg.pixmax)==2
    pixmin = arg.pixmax(1);
    pixmax = arg.pixmax(2);
elseif length(arg.pixmax)==1
    pixmin = 0;
    pixmax = arg.pixmax;
else
    error pixmax
end

% likelihood denom, if not provided
denom = arg.denom;
if isempty(denom)
    denom = Ab'*col(arg.aai.*wi); % requires real a_ij and a_ij >= 0
end
if ~arg.update_even_if_denom_0
    denom(denom==0) = inf; % trick: prevents pixels where denom=0 being updated
end

if isempty(R)
    pgrad = 0; % unregularized default
    Rdenom = 0;
end

if isnumeric(arg.rho)
    if arg.rho>0
        rho = str2func(['@(k)' num2str(arg.rho)]);
    else
        fail 'illegal al penalty parameter rho?!';
    end
else
    rho = str2func(arg.rho);
end

if isnumeric(arg.alpha)
    if arg.alpha>=0
        alpha = str2func(['@(k)' num2str(arg.alpha)]);
    else
        fail 'illegal relaxation parameter alpha?!';
    end
else
    alpha = str2func(arg.alpha);
end

[nb,na] = size(yi);

x = x(:);
np = length(x);
xs = zeros(np,length(arg.isave));
if any(arg.isave==0)
    xs(:,arg.isave==0) = x;
end

%info = zeros(niter,?); % do not initialize since size may change

% initilization
iblock = nblock;
ia = iblock:nblock:na;
li = Ab{iblock}*x;
li = reshape(li,nb,length(ia));
resid = wi(:,ia).*(li-yi(:,ia));
if arg.scale_nblock
    scale = nblock; % traditional way
else
    scale = na/numel(ia); % alternative - untested
end
grad = scale*Ab{iblock}'*resid(:);

g = grad;
h = denom.*x-grad;

k = 1;

% iterate
for iter = 1:arg.niter
    ticker(mfilename,iter,arg.niter)

    relax = relax0/(1+relax_rate*(iter-1));

    % loop over subsets
    for iset = 1:nblock
        % update image x using fista
        z = x; v = z;
        zold = z; told = 1;
        s = rho(k)*grad+(1-rho(k))*g;
        for ifiter = 1:arg.nfiter
            sigma = rho(k)*denom.*(v-x)+s;
            if ~isempty(R)
                pgrad = R.cgrad(R,v);
                Rdenom = R.denom(R,v);
            end
            
            num = sigma+pgrad;
            den = rho(k)*denom/relax+Rdenom;
            
            z = v-num./den;
            z = max(z,pixmin);
            z = min(z,pixmax);
            
            % adaptive restart
            if (v-z)'*(z-zold)>0
                t = 1;
                v = z;
            else
                t = (1+sqrt(1+4*told^2))/2;
                v = z+(t-1)/told*(z-zold);
            end
            
            zold = z; told = t;
        end
        x = z;
        
        iblock = starts(iset);
        ia = iblock:nblock:na;

        li = Ab{iblock}*x;
        li = reshape(li,nb,length(ia));
        resid = wi(:,ia).*(li-yi(:,ia));
        
        if arg.scale_nblock
            scale = nblock; % traditional way
        else
            scale = na/numel(ia); % alternative - untested
        end
        
        grad = scale*Ab{iblock}'*resid(:); % A' * W * (y - A*x)
        
        g = (rho(k)*(alpha(k)*grad+(1-alpha(k))*g)+g)/(rho(k)+1);
        
        k = k+1;
    end

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
