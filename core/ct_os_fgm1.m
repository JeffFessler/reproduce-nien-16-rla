  function [x, info, relax_iter] = ct_os_fgm1(x0, Ab, yi, R, varargin)
%|function [x, info, relax_iter] = ct_os_fgm1(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using fast gradient
%| method [kim:14:ofo] with (optionally relaxed) ordered subsets
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
%|    type_mom                type of momentum (default: 'std')
%|                            options: 'std' | 'opt'
%|    type_reg                type of regularizer (default: 'huber')
%|                            options: 'max' | 'huber' | 'prox'
%|    step_thd0               initial step threshold (default: 0, i.e., no relax)
%|    roi                     region-of-interest
%|
%| out
%|    x        [nx ny nz]     reconstructed image
%|    info     [niter 1]      outcome of user defined function
%|    relax_iter              relax iterations
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
arg.type_mom = 'std';
arg.type_reg = 'huber';
arg.step_thd0 = 0;
arg.roi = [];
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
    fail 'bad relax0';
end

% voxel value limits
if length(arg.voxmax)==2
    voxmin = arg.voxmax(1);
    voxmax = arg.voxmax(2);
elseif length(arg.voxmax)==1
    voxmin = 0;
    voxmax = arg.voxmax;
else
    fail 'bad voxmax';
end
proj = @(x) max(min(x,voxmax),voxmin);

% likelihood denom, if not provided
Lden = arg.denom;
if isempty(Lden)
    Lden = Ab'*(wi(:).*(Ab*ones(np,1,'single')));
end
Lden(Lden==0) = inf;

switch arg.type_mom
    case 'std'
        is_opt = false;
    case 'opt'
        is_opt = true;
    otherwise
        fail 'bad type_mom';
end

switch arg.type_reg
    case 'max'
        is_huber = false;
        is_prox = false;
        Rden = R.denom(R,ones(np,1,'single'));
    case 'huber'
        is_huber = true;
        is_prox = false;
    case 'prox'
        is_prox = true;
    otherwise
        fail 'bad type_reg';
end

if isempty(arg.roi)
    roi = ones(np,1,'single');
else
    roi = arg.roi;
end

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count,back] = loop_count_str(nblock);

x = tov(x0);
z = x;
t = 1;
xold = x;
told = t;

xp = x;
thd = arg.step_thd0;
relax = relax0;
relax_iter = [];

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using OS-FGM1 (nblock = %g, type_mom = %s, type_reg = %s)...\\n',nblock,arg.type_mom,arg.type_reg);
fprintf(str.info);
str.log = str.info;

for iter = 1:niter
    tic;
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        ib = ios(iset);
        ia = ib:nblock:na;
        num = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*z-col(yi(:,:,ia)))));
        
        if is_prox
            den = Lden/relax;
            x = edge_shrink(z-num./den,R,den,'voxmax',[voxmin voxmax]);
        else
            num = num+R.cgrad(R,z);
            if is_huber
                den = Lden+R.denom(R,z);
            else
                den = Lden+Rden;
            end
            x = proj(z-relax*num./den);
        end
        
        t = (1+sqrt(1+4*told^2))/2;
        if is_opt
            z = x+(told-1)/t*(x-xold)+told/t*(x-z);
        else
            z = x+(told-1)/t*(x-xold);
        end
        xold = x;
        told = t;
        
        fprintf([back count],iset);
    end
    
    tt = toc;
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf(' ');
    str.info = sprintf('info = %g (%g: in %g seconds)',info(iter),iter,tt);

    if norm((x-xp).*roi)/sqrt(sum(roi))<thd/2
        relax = relax0/(1+relax_rate*nblock*iter);
        thd = thd/2;
        relax_iter = [relax_iter iter];
        str.info = [str.info '... relax!'];
    end
    xp = x;   % update xp

    str.info = [str.info '\n'];
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
