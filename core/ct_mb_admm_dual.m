  function [x, info] = ct_mb_admm_dual(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_mb_admm_dual(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using multi-block ADMM (dual)
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
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default: {})
%|    is_we                   weighte-embedded or not (default: 1)
%|    is_or                   over-relaxation is used or not (default: 0)
%|    rho                     al penalty parameter (default: 1)
%|    nditer                  # of image denoise iterations (default: 50)
%|    is_stoc                 random access or not (default: 0)
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
arg.is_we = 1;
arg.is_or = 0;
arg.rho = 1;
arg.nditer = 50;
arg.is_stoc = 0;
arg = vararg_pair(arg,varargin);

% make directory if it dose not exist
if ~exist(arg.path,'dir')
  mkdir(arg.path);
end

% local parameter
niter = arg.niter;
rho = arg.rho;
nditer = arg.nditer;
is_we = arg.is_we;
is_or = arg.is_or;
rate = 0.01;

% system matrix
Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');
np = Ab.np;
% na = Ab.arg.odim(end);
% [ns,nt,na] = Ab.odim;
ns = Ab.odim(1);
nt = Ab.odim(2);
na = Ab.odim(3);
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

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count,back] = loop_count_str(nblock);
[count1,back1] = loop_count_str(nditer);

% initialize multi-block admm (dual)
% ----------------------------------------- image domain
x = tov(x0);               % recon image (dual)
v = zeros(np,1,'single');  % accumulator
u = v;                     % auxiliary variable
% ------------------------------------ projection domain
z = cell(1,nblock);
d = cell(1,nblock);
nd = zeros(nblock,1);
tic;
fprintf(['compute diagonal majorizer ' count],0);
for ib = 1:nblock
    ia = ib:nblock:na;
    nd(ib) = ns*nt*length(ia);
    z{ib} = zeros(nd(ib),1,'single');  % dual image (primal)
    % diagonal majorizer
    if is_we
        sqrt_wb = sqrt(col(wi(:,:,ia)));
        d{ib} = sqrt_wb.*(Ab{ib}*(Ab{ib}'*sqrt_wb));
    else
        d{ib} = Ab{ib}*(Ab{ib}'*ones(nd(ib),1,'single'));
    end
    fprintf([back count],ib);
end
tt = toc;
fprintf(' (elapsed time is %.3f seconds)\n',tt);

if arg.is_stoc
    block_access = randperm1(nblock,niter);  % without replacement
else
    block_access = repmat(subset_start(nblock)',[niter 1]);
end
% block_access

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);
rho0 = rho;

str.info = sprintf('solve x-ray ct image recon using mb-admm (nblock = %d, rho0 = %f, rate = %f)...\\n',nblock,rho0,rate);
fprintf(str.info);
str.log = str.info;
for iter = 1:niter
    tic;
    
    iset = 0;
    rho = rho0*(1+rate*(iter-1));
    fprintf(['block update ' count],iset);
    for ib = block_access(iter,:)
        % z-update
        ia = ib:nblock:na;
        den = rho*d{ib};
        if ~is_we, den = den.*col(wi(:,:,ia)); end
        den = 1+den;
        r = col(yi(:,:,ia))-rho*Ab{ib}*(v-u+x/rho);
        if is_we, r = sqrt(col(wi(:,:,ia))).*r; end
        znew = (r+rho*d{ib}.*z{ib})./den;
        if ~is_we, znew = col(wi(:,:,ia)).*znew; end
        % v-update
        dz{ib} = znew-z{ib};
        if is_we
            v = v+Ab{ib}'*(sqrt(col(wi(:,:,ia))).*dz{ib});
        else
            v = v+Ab{ib}'*dz{ib};
        end        
        z{ib} = znew;
        iset = iset+1;
        fprintf([back count],iset);
    end
    % x-update
    xnew = x;
    w = xnew;
    xold = xnew;
    told = 1;
    diter = 0;
    fprintf([' -> image denoise ' count1],0);
    for diter = 1:nditer
        num = (w-(x+rho*v))+rho*R.cgrad(R,w);
        den = 1+rho*R.denom(R,w);
        xnew = w-num./den;
        xnew = max(xnew,voxmin);
        xnew = min(xnew,voxmax);
        rel_step = norm(xnew-xold)/norm(xold);
        
        if (w-xnew)'*(xnew-xold)>0
            t = 1;
            w = xnew;
        else
            t = (1+sqrt(1+4*told^2))/2;
            w = xnew+(told-1)/t*(xnew-xold);
        end
        xold = xnew;
        told = t;
        fprintf([back1 count1],diter);
    end
    % u-update
    uold = u;
    u = v+(x-xnew)/rho;
    if is_or
        % x = x+1.618*rho*(v-u);
        x = x+1.618*(xnew-x);
    else
        x = xnew;
    end
    
    tt = toc;
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf(' ');
    str.info = sprintf('info(%d) = %.3f (elapsed time is %.3f seconds)\\n',iter,info(iter),tt);
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
