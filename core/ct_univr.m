  function [x, info] = ct_univr(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_univr(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using a universal variance
%| reduction framework proposed in zhu-15-uau
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
%|    naa                     # of forward/back-projections (default: 10)
%|    m0                      # of stoc grad evals at the 0th iter (default: nblock)
%|    step_type               type of step size (default: 'max')
%|    isave                   indices of images to be saved (default: [])
%|    path                    path to saved images (default: './')
%|    wi       [ns nt na]     weighting sinogram (default: [] for uniform)
%|    voxmax   [1] or [2]     max voxel value, or [min max] (default: [0 inf])
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|
%| out
%|    x        [nx ny nz]     reconstructed image
%|    info     [niter 1]      outcome of user defined function
%|

if nargin<4, help(mfilename), error(mfilename), end

% block data terms
Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nb_data = block_op(Ab,'n');

% block reg terms
fprintf('make block reg terms...\n');
nb_reg = R.M;
r = cell(1,nb_reg);
for ir = 1:nb_reg
    r{ir} = Reg1(...
    R.data.cdp_arg{1},... % kappa
    'offsets_is_zxy',R.offsets_is_zxy,...
    'offsets',R.data.offsets(ir),...
    'beta',R.data.beta(ir),...
    'type_penal','mex',...
    'nthread',jf('ncore'),...
    'distance_power',R.data.distance_power,...
    'mask',R.data.mask,...
    'pot_arg',R.data.pot_arg{1}...
    );
end

% number of blocks
nblock = nb_data+nb_reg;

% defaults
arg.naa = 10;
arg.m0 = nblock;
arg.step_type = 'max';
arg.isave = [];
arg.path = './';
arg.userfun = @userfun_default;
arg.userarg = {};
arg.voxmax = inf;
arg.wi = [];
arg = vararg_pair(arg,varargin);

if ~exist(arg.path,'dir')
  mkdir(arg.path);
end

naa = arg.naa;
m0 = arg.m0;
step_type = arg.step_type;
log = [ ...
    'x-ray ct image reconstruction using the univr framework...\n' ...
    '==========================================================\n' ...
    'parameter list:\n' ...
    'naa: ' num2str(naa) '\n' ...
    'm0: ' num2str(m0) '\n' ...
    'step_type: ' step_type '\n' ...
    '==========================================================\n' ...
    ];

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

% compute denom
fprintf('compute denom... ');
% denom = 0;
% [count,back] = loop_count_str(nb_data);
% fprintf(count,0);
% for ib = 1:nb_data
%     ia = ib:nb_data:na;
%     den = Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*ones(np,1)));
%     if strcmp(arg.step_type,'max')
%         denom = max(den,denom);
%     elseif strcmp(arg.step_type,'mean')
%         denom = denom+den/nblock;
%     else
%         fail denom;
%     end
%     fprintf([back count],ib);
% end
% denom(denom==0) = inf;
% fprintf(' ');
% [count,back] = loop_count_str(nb_reg);
% fprintf(count,0);
% for ir = 1:nb_reg
%     den = r{ir}.denom(r{ir},ones(np,1));
%     if strcmp(arg.step_type,'max')
%         denom = max(den,denom);
%     elseif strcmp(arg.step_type,'mean')
%         denom = denom+den/nblock;
%     else
%         fail denom;
%     end
%     fprintf([back count],ir);
% end
fprintf('\n');
% denom = Ab'*(wi(:).*(Ab*ones(np,1,'single')))/nb_data;
% denom(denom==0) = inf;
% denom = max(denom,R.denom(R,ones(np,1,'single'))/nb_reg);
% denom = denom*7;
denom = Ab'*(wi(:).*(Ab*ones(np,1,'single')))/nb_data*nblock;
denom(denom==0) = inf;
denom = max(denom,R.denom(R,ones(np,1,'single')/nb_reg*nblock));
denom = denom*7;

% if any(arg.isave==0)
%     fld_write([arg.path 'x_iter_0.fld' ],x0);
% end

% initialize the algorithm
x = min(max(tov(x0),voxmin),voxmax);
xt = x;

ep = 0;
ds = arg.userfun(x,arg.userarg{:});
fprintf('epoch %g: info = %f\n',ep(end),ds(end));

epoch = 0;
s = 1;
fprintf('start solving x-ray ct image reconstruction using the universal variance reduction framework...\n');
while epoch<naa
    gt = Ab'*(wi(:).*(Ab*xt-yi(:)))+R.cgrad(R,xt);
    epoch = epoch+1;
    fprintf('epoch %g: info = %f\n',epoch,arg.userfun(x,arg.userarg{:}));
    
    ms = 2^s*m0;
    xs = 0;
    for t = 1:ms
        ib = randi(nblock);
        if ib<=nb_data % pick a data term
            ia = ib:nb_data:na;
            zeta = nblock*Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*(x-xt)))+gt;
            epoch = epoch+1/nb_data;
        elseif ib>nb_data % pick a reg term
            ir = ib-nb_data;
            zeta = nblock*(r{ir}.cgrad(r{ir},x)-r{ir}.cgrad(r{ir},xt))+gt;
        end
        x = min(max(x-zeta./denom,voxmin),voxmax);
        if mod(t,ceil(m0/10))==0
            fprintf('epoch %g: info = %f\n',epoch,arg.userfun(x,arg.userarg{:}));
        end
        
        xs = xs+x/ms;
    end
    xt = xs;
end

x = tom(x);

fid = fopen([arg.path 'recon.log'],'wt');
fprintf(fid,log);
fclose(fid);
