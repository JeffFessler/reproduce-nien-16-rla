  function [x, info] = ct_sdca_sag1(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_sdca_sag1(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using (mini-batch)
%| stochastic dual coordinate ascent and stochastic average gradient
%| (low memory)
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
%|    denom    [1 nblock]     precomputed denominators
%|    voxmax   [1] or [2]     max voxel value, or [min max] (default: [0 inf])
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    psag                    probability of sag update (default: 0.1)
%|    nreg                    # of sub-regularizers (default: 1)
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
arg.psag = 0.1;
arg.nreg = 1;
arg = vararg_pair(arg,varargin);

Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');
ios = subset_start(nblock);
np = Ab.np;
nd = Ab.nd;
na = Ab.odim(end);
tov = @(z) z(Ab.imask);
tom = @(z) embed(z,Ab.imask);
niter = arg.niter;

x = tov(x0);
dreg = R.denom(R,zeros(np,1,'single'))+eps;

% progress status
[count,back] = loop_count_str(nblock);

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

% likelihood denom, if not provided
denom = arg.denom;
if isempty(denom)
    fprintf(['Start computing denominator... ' count],0);
    denom = cell(1,nblock);
    for iblock = 1:nblock
        denom{iblock} = Ab{iblock}*((Ab{iblock}'*ones(Ab{iblock}.size(1),1,'single'))./dreg);
        fprintf([back count],iblock);
    end
    fprintf('\n');
else
    if length(denom)~=nblock
        fail 'bad denom size';
    end
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

% probability of sag update
if isnumeric(arg.psag)
    if arg.psag>0 && arg.psag<=1
        psag = arg.psag;
    else
        fail 'bad psag';
    end
else
    fail 'bad psag';
end

% number of sub-regularizers
if isnumeric(arg.nreg)
    if arg.nreg>=1 && arg.psag<=R.M
        nreg = arg.nreg;
    else
        fail 'bad nreg';
    end
else
    fail 'bad nreg';
end

% compute majorizing matrix of sinogram update
dreg = R.denom(R,zeros(np,1,'single'))+eps;
dsin = cell(1,nblock);
z = cell(1,nblock);
for iblock = 1:nblock
    ia = iblock:nblock:na;
    dsin{iblock} = denom{iblock}.*col(wi(:,:,ia))+1;
    z{iblock} = zeros(Ab{iblock}.size(1),1);
end

% prepare sub-regularizers
[Rs,exslab,inslab] = Reg1slab(R,nreg);
sgx = R.cgrad(R,x);

% initialization
b = x-sgx./dreg;
awz = zeros(np,1);

x = b-awz./dreg;
x = max(x,voxmin);
x = min(x,voxmax);

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

str.info = sprintf('Start solving X-ray CT image reconstruction problem using SDCA-SAG (low-mem)...\\n',nblock);
fprintf(str.info);
str.log = str.info;

for iter = 1:niter
    tic;
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        iblock = randi(nblock);
        
        ia = iblock:nblock:na;
        dz = (Ab{iblock}*x-col(yi(:,:,ia))-z{iblock})./dsin{iblock};
        z{iblock} = z{iblock}+dz;
        awz = awz+Ab{iblock}'*(col(wi(:,:,ia)).*dz);
        
        x = b-awz./dreg;
        x = max(x,voxmin);
        x = min(x,voxmax);
        
        % update reg majorizer using sag
        if rand<=psag
            ir = randi(nreg);
            sgx = inslab{ir}(Rs{ir}.cgrad(Rs{ir},exslab{ir}(x)),sgx);
            b = x-sgx./dreg;
        end
        
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
