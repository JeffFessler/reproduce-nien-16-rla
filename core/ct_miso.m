  function [x, info] = ct_miso(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_miso(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using minimization by
%| incremental surrogate optimization (miso).
%|
%| cost(x) = (y-Ax)'W(y-Ax)/2+R(x)
%|
%| in
%|    x0         [nx ny nz]     initial estimate
%|    Ab         [nd np]        Gblock object from Gcone
%|    yi         [ns nt na]     measurements (noisy sinogram data)
%|    R                         penalty object (see Reg1.m)
%|
%| option
%|    niter                     # of iterations (default: 1)
%|    isave                     indices of images to be saved (default: [])
%|    path                      path to saved images (default: './')
%|    wi         [ns nt na]     weighting sinogram (default: [] for uniform)
%|    voxmax     [1] or [2]     max voxel value, or [min max] (default: [0 inf])
%|    denom      [np 1]         precomputed denominator
%|    relax0     [1] or [2]     relax0 or (relax0, relax_rate)
%|    userfun    @              user defined function handle (see default below)
%|                              taking arguments (x, userarg{:})
%|    userarg    {}             user arguments to userfun (default {})
%|    type_reg                  type of regularizer (default: 'max')
%|                              options: 'max' and 'prox'
%|    use_miso1                 use miso1 heuristic (default: 0)
%|
%| out
%|    x          [nx ny nz]     reconstructed image
%|    info       [niter 1]      outcome of user defined function
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
arg.type_reg = 'max';
arg.use_miso1 = 0;
arg = vararg_pair(arg,varargin);

if ~exist(arg.path,'dir')
  mkdir(arg.path);
end

% system matrix
Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');
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
denom0 = denom;
denom(denom==0) = inf;

% regularization
type_reg = arg.type_reg;
switch type_reg
    case 'max'
        is_prox = 0;
        den = denom+R.denom(R,ones(np,1,'single'));
    case 'prox'
        is_prox = 1;
        den = denom;
    otherwise
        fail 'bad type_reg';
end
if arg.use_miso1
    den = den*0.05;
end

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count,back] = loop_count_str(nblock);

% initialize miso
x = tov(x0);
alp = x(:,ones(1,nblock));
mu = x;
info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);
access = randi(nblock,[niter nblock]);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using MISO (nblock = %g)...\\n',nblock);
fprintf(str.info);
str.log = str.info;

for iter = 1:niter
    tic;
    
    relax = relax0/(1+relax_rate*(iter-1));
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        % compute gradient
        ib = access(iter,iset);
        ia = ib:nblock:na;
        grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*x-col(yi(:,:,ia)))));
        
        % update historic data
        if is_prox
            num = grad;
        else
            num = grad+R.cgrad(R,x);
        end
        alp_new = x-relax*num./den;
        mu = mu+(alp_new-alp(:,ib))/nblock;
        alp(:,ib) = alp_new;
        
        % update image
        if is_prox
            x = edge_shrink1(mu,R,relax,den,'voxmax',[voxmin voxmax]);
        else
            x = mu;
        end
        x = max(x,voxmin); % project image no matter what reg is used
        x = min(x,voxmax);
        
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
