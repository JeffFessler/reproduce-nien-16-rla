  function [x, info] = ct_fgm_rs(x0, A, yi, R, varargin)
%|function [x, info] = ct_fgm_rs(x0, A, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using fast gradient
%| method [kim:14:ofo] with restart
%|
%| cost(x) = (y-Ax)'W(y-Ax)/2+R(x)
%|
%| in
%|    x0       [nx ny nz]     initial estimate
%|    A        [nd np]        Gcone object
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
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
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
arg.denom = [];
arg = vararg_pair(arg,varargin);

np = A.np;
if A.zxy==0 && R.offsets_is_zxy==0
    tov = @(x) masker(x,A.imask); % 3d (xyz) to 1d (xyz)
    tom = @(x) embed(x,A.imask); % 1d (xyz) to 3d (xyz)
elseif A.zxy==1 && R.offsets_is_zxy==1
    tov = @(x) masker(permute(x,[3 1 2]),A.imask); % 3d (xyz) to 1d (zxy)
    tom = @(x) permute(embed(x,A.imask),[2 3 1]); % 1d (zxy) to 3d (xyz)
else
    fail 'mismatched order of A and R';
end
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
    denom = A'*(wi(:).*(A*ones(np,1)));
end

if ~isnumeric(denom)
    error denom;
end
denom(denom==0) = inf;

x = tov(x0);
z = x;
xold = x;
told = 1;

dr = R.denom(R,ones(np,1,'single'));
den = denom+dr+eps;

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using FGM with restart...\\n');
fprintf(str.info);
str.log = str.info;

for iter = 1:niter
    fprintf(['iter ' num2str(iter) ': ']);
    tic;
    
    num = A'*(wi(:).*(A*z-yi(:)))+R.cgrad(R,z);
    den = denom+R.denom(R,z)+eps;   % change to huber's curvature due to the sparse-view experiment
    
    x = z-num./den;
    x = max(x,voxmin);
    x = min(x,voxmax);
    
    if (z-x)'*(x-xold)>0
        t = 1;
        z = x;
    else
        t = (1+sqrt(1+4*told^2))/2;
        z = x+(told-1)/t*(x-xold);
    end
    xold = x;
    told = t;
	
    tt = toc;
    info(iter) = arg.userfun(x,arg.userarg{:});
    str.info = sprintf('info = %g (%g: in %g seconds)\\n',info(iter),iter,tt);
    fprintf(str.info);
    str.log = strcat(str.log,str.info);
    
    if any(arg.isave==iter)
        fld_write([arg.path 'x_iter_' num2str(iter) '.fld' ],tom(x));
    end
end

x = tom(x);
info = [info0; info];

fid = fopen([arg.path 'recon.log'],'wt');
fprintf(fid,str.log);
fclose(fid);
