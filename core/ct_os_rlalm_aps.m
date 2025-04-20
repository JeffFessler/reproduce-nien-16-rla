  function [x, info] = ct_os_rlalm_aps(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_os_rlalm_aps(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using relaxed linearized
%| augmented lagrangian method with (optionally relaxed) ordered subsets
%| and adaptive parameter scaling
%|
%| cost(x) = (y-Ax)'W(y-Ax)/2+R(x)
%|
%| in
%|    x0       [nx ny nz]     initial estimate
%|    Ab       [nd np]        Gblock object from Gcone
%|    yi       [ns nt na]     measurements (noisy sinogram data)
%|    R                       penalty object (see Reg1.m), can be []
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    isave                   indices of images to be saved (default: [])
%|    path                    path to saved images (default: './')
%|    wi       [ns nt na]     weighting sinogram (default: [] for uniform)
%|    voxmax   [1] or [2]     max voxel value, or [min max] (default: [0 inf])
%|    denom    [np 1]         precomputed denominator
%|    relax0   [1] or [2]     relax0 or (relax0, relax_rate)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    rho0                    initial al penalty parameter (default: 1)
%|    mu0                     acceptable primal/dual residual ratio (default: 10)
%|    tau                     al penalty parameter adaptation parameter (default: 2)
%|    beta                    acceptable ratio adaptation parameter (default: 1.1)
%|    thd                     primal/dual adaptation threshold (default: 0)
%|    alpha                   relaxation parameter (default: 1)
%|
%| out
%|    x        [nx ny nz]     reconstructed image
%|    info     [niter 1]      outcome of user defined function
%|

if nargin<4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.isave = [];
arg.path = './';
arg.userfun = @userfun_default;
arg.userarg = {};
arg.voxmax = inf;
arg.wi = [];
arg.relax0 = 1;
arg.denom = [];
arg.rho0 = 1;
arg.mu0 = 10;
arg.tau = 2;
arg.beta = 1.1;
arg.thd = 0;
arg.alpha = 1;
arg = vararg_pair(arg,varargin);

Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');
ios = subset_start(nblock);
np = Ab.np;
na = Ab.odim(end);
tov = @(z) z(Ab.imask);
tom = @(z) embed(z,Ab.imask);
niter = arg.niter;

% statistical weighting matrix
wi = arg.wi;
if isempty(wi)
    wi = ones(size(yi));
end

% check input sinogram sizes for OS
if ndims(yi)~=3 || (size(yi,3)==1 && nblock>1)
    fail 'bad yi size';
end
if ndims(wi)~=3 || (size(wi,3)==1 && nblock>1)
    fail 'bad wi size';
end

% relaxed ordered-subsets
relax0 = arg.relax0(1);
if length(arg.relax0)==1
    relax_rate = 0;
elseif length(arg.relax0)==2
    relax_rate = arg.relax0(2);
else
    error relax;
end

% voxel value limits
if length(arg.voxmax)==2
    voxmin = arg.voxmax(1);
    voxmax = arg.voxmax(2);
elseif length(arg.voxmax)==1
    voxmin = 0;
    voxmax = arg.voxmax;
else
    error voxmax;
end

% likelihood denom, if not provided
denom = arg.denom;
if isempty(denom)
    denom = Ab'*(wi(:).*(Ab*ones(np,1)));
end

if ~isnumeric(denom)
    if strcmp(denom,'max')
        denom = 0;
        for ib = 1:nblock
            ia = ib:nblock:na;
            denom = max(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*ones(np,1))),denom);
        end
        denom = denom*nblock;
    else
        error denom;
    end
end
denom(denom==0) = inf;

% relaxation parameter
if isnumeric(arg.alpha)
    if arg.alpha>=0
        alpha = str2func(['@(k)' num2str(arg.alpha)]);
    else
        fail 'illegal relaxation parameter alpha?!';
    end
else
    alpha = str2func(arg.alpha);
end

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count,back] = loop_count_str(nblock);

x = tov(x0);
ib = ios(end);
ia = ib:nblock:na;
grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*x-col(yi(:,:,ia)))));

g = grad;
h = denom.*x-grad;

rho = arg.rho0;
sqrtL = sqrt(median(denom));
ddd = denom; ddd(ddd==inf) = 0;

mu = arg.mu0;

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

ppr = zeros(niter*nblock,1);
ddr = zeros(niter*nblock,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-rLALM with APS (nblock = %g)...\\n',nblock);
fprintf(str.info);
str.log = str.info;

for iter = 1:niter
    tic;
    
    relax = relax0/(1+relax_rate*(iter-1));
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        k = (iter-1)*nblock+iset;
        
        xold = x; gradold = grad; gold = g;
        
        num = rho*denom.*x+(1-rho)*g-rho*h+R.cgrad(R,x);
        den = rho*denom/relax+R.denom(R,x);

        x = x-num./den;
        x = max(x,voxmin);
        x = min(x,voxmax);
        
        ib = ios(iset);
        ia = ib:nblock:na;
        grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*x-col(yi(:,:,ia)))));
        
        g = (rho*(alpha(k)*grad+(1-alpha(k))*g)+g)/(rho+1);
        h = alpha(k)*(denom.*x-grad)+(1-alpha(k))*h;
        
        pr = norm(grad-g)/sqrtL;
        dr = norm((g-gold)-(grad-gradold)+ddd.*(x-xold))*rho;
        
        if max(pr,dr)>arg.thd
            if pr>mu*dr
                rho = rho*arg.tau;
                mu = mu*arg.beta;
            elseif dr>mu*pr
                rho = rho/arg.tau;
                mu = mu*arg.beta;
            end
        end

        ppr(k) = pr;
        ddr(k) = dr;
        
        fprintf([back count],iset);
    end
    
    tt = toc;
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf(' ');
    str.info = sprintf('rho = %g, info = %g (%g: in %g seconds)\\n',rho,info(iter),iter,tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    
    if any(arg.isave==iter)
        fld_write([arg.path 'x_iter_' num2str(iter) '.fld' ],tom(x));
    end
end

x = tom(x);
info = [info0; info];

figure; semilogy([ppr ddr]);

fid = fopen([arg.path 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
