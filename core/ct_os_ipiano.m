  function [x, info] = ct_os_ipiano(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_os_ipiano(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using ipiano
%| (see ochs-14-iip)
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
%|    niter                   # of iterations (default: 10)
%|    isave                   indices of images to be saved (default: [])
%|    path                    path to saved images (default: './')
%|    wi       [ns nt na]     weighting sinogram (default: [] for uniform)
%|    voxmax   [1] or [2]     max voxel value, or [min max] (default: [0 inf])
%|    denom    [np 1]         precomputed denominator
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    beta                    the momentum parameter (default: @(k)0.75)
%|
%| out
%|    x        [nx ny nz]     reconstructed image
%|    info     [niter 1]      outcome of user defined function
%|

if nargin<4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 10;
arg.isave = [];
arg.path = './';
arg.userfun = @userfun_default;
arg.userarg = {};
arg.voxmax = inf;
arg.wi = [];
arg.denom = [];
arg.beta = @(k)0.75;
arg = vararg_pair(arg,varargin);

if ~exist(arg.path,'dir')
  mkdir(arg.path);
end

Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');
ios = subset_start(nblock);
np = Ab.np;
na = Ab.arg.odim(end);
if Ab.zxy==0 && R.offsets_is_zxy==0
    tov = @(x) masker(x,Ab.imask); % 3d (xyz) to 1d (xyz)
    tom = @(x) embed(x,Ab.imask); % 1d (xyz) to 3d (xyz)
elseif Ab.zxy==1 && R.offsets_is_zxy==1
    tov = @(x) masker(permute(x,[3 1 2]),Ab.imask); % 3d (xyz) to 1d (zxy)
    tom = @(x) permute(embed(x,Ab.imask),[2 3 1]); % 1d (zxy) to 3d (xyz)
else
    fail 'mismatched order of Ab and R';
end

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
proj = @(x) min(max(x,voxmin),voxmax);

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

niter = arg.niter;
beta = arg.beta;

log = [ ...
    'x-ray ct image reconstruction using os-ipiano...\n' ...
    '==========================================================\n' ...
    'number of iterations: ' num2str(niter) '\n' ...
    'number of blocks: ' num2str(nblock) '\n' ...
    'beta: ' func2str(beta) '\n' ...
    '==========================================================\n' ...
    ];
fprintf(log);

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count,back] = loop_count_str(nblock);

x = proj(tov(x0));
xp = x;
xpp = x;

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

for iter = 1:niter
    tic;
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        k = iset+nblock*(iter-1);
        
        ib = ios(iset);
        ia = ib:nblock:na;
        grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*xp-col(yi(:,:,ia)))));
        
        num = grad+R.cgrad(R,xp);
        den = denom+R.denom(R,xp);
        x = proj(xp-2*(1-beta(k))*num./den+beta(k)*(xp-xpp));
        
        xpp = xp;
        xp = x;
        
        fprintf([back count],iset);
    end
    
    iter_time = toc;
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf(' ');
    str = sprintf('info = %g (%g: in %g seconds)\\n',info(iter),iter,iter_time);
    fprintf(str);
    log = strcat(log,str);
    
    if any(arg.isave==iter)
        fld_write([arg.path 'x_iter_' num2str(iter) '.fld' ],tom(xcur));
    end
end

x = tom(xcur);
info = [info0; info];

fid = fopen([arg.path 'recon.log'],'wt');
fprintf(fid,log);
fclose(fid);
