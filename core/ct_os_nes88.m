  function [x, info] = ct_os_nes88(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_os_nes88(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using Nesterov's second
%| fast gradient method with (optionally relaxed) ordered subsets
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
%|    average                 averaged ordered subsets (default: 'no')
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
arg.average = 'no';
arg = vararg_pair(arg,varargin);

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

% averaged ordered subsets
switch arg.average
    case {'no','yes'}
        avg_os = strcmp(arg.average,'yes');
    otherwise
        fail 'illegal average option';
end

% progress status
[count,back] = loop_count_str(nblock);

x = tov(x0); x = max(x,voxmin); x = min(x,voxmax);
z = x;
ib = ios(nblock);
ia = ib:nblock:na;
if avg_os
    gx = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*x-col(yi(:,:,ia)))));
    gz = gx;
end
t = 1;

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-Nes88 (nblock = %g, average = %s)...\\n',nblock,arg.average);
fprintf(str.info);
str.log = str.info;

for iter = 1:niter
    tic;
    
    relax = relax0/(1+relax_rate*(iter-1));
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        ib = ios(iset);
        ia = ib:nblock:na;
        k = nblock*(iter-1)+iset;
        % theta = 2/(k+1);
        theta = 1/t;
        
        xb = (1-theta)*z+theta*x;
        if avg_os
            gxb = (1-theta)*gz+theta*gx;
        else
            gxb = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*xb-col(yi(:,:,ia)))));
        end
        
        num = gxb+R.cgrad(R,xb);
        % den = theta*(denom+R.denom(R,xb));
        den = theta*denom+R.denom(R,xb);
        x = x-relax*num./den;
        x = max(x,voxmin);
        x = min(x,voxmax);
        
        z = (1-theta)*z+theta*x;
        if avg_os
            gx = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*x-col(yi(:,:,ia)))));
            gz = (1-theta)*gz+theta*gx;
        end

        t = (1+sqrt(1+4*t^2))/2;
        
        fprintf([back count],iset);
    end
    
    tt = toc;
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf(' ');
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
