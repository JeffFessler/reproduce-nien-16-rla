  function [x, info] = ct_os_ppm0(x0, Ab, yi, R, varargin)
%|function [x, info] = ct_os_ppm0(x0, Ab, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using (fast) proximal point
%| method with inner fast os algorithm
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
%|    inner_meth              inner loop method (default: 'fgm')
%|                            method: 'fgm2', 'ogm2', 'lalm', 'rlalm'
%|    outer_meth              outer loop method (default: {'nes83',@(k)2/(k+1)})
%|                            method: 'nes83', 'nes88' (w/ bug)
%|    t                       proximal point step size (default: mean(denom))
%|    period                  ppm update period (default: 2)
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
arg.niter = 10;
arg.inner_meth = 'fgm';
arg.outer_meth = {'nes83',@(k) 2/(k+1)};
arg.t = [];
arg.period = 2;
arg.isave = [];
arg.path = './';
arg.userfun = @userfun_default;
arg.userarg = {};
arg.voxmax = inf;
arg.wi = [];
arg.relax0 = 1;
arg.denom = [];
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
denom0 = denom;
denom(denom==0) = inf;

% proximal point step size
t = arg.t;
if isempty(t)
    t = mean(denom0);
end

niter = arg.niter;
period = arg.period;
inner_meth = arg.inner_meth;
outer_meth = arg.outer_meth{1};
theta = arg.outer_meth{2};

log = [ ...
    'x-ray ct image reconstruction using os-ppm...\n' ...
    '==========================================================\n' ...
    'number of iterations: ' num2str(niter) '\n' ...
    'number of blocks: ' num2str(nblock) '\n' ...
    'inner method: ' inner_meth '\n' ...
    'outer method: ' outer_meth '\n' ...
    'theta: ' func2str(theta) '\n' ...
    'proximal point step size: ' num2str(t) '\n' ...
    'proximal point update period: ' func2str(period) '\n' ...
    '==========================================================\n' ...
    ];
fprintf(log);

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count,back] = loop_count_str(nblock);

x = proj(tov(x0));
xold = x;
ip = 1;
switch outer_meth
    case 'nes83'
        v = x+theta(ip)*(1-theta(ip-1))/theta(ip-1)*(x-xold);
        gain = t;
    case 'nes88'
        v = x;
        gain = t/theta(ip);
    otherwise
        error outer_meth;
end
xcur = x;
u = xcur;
ii = 1;
ib = ios(end);
ia = ib:nblock:na;
grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*u-col(yi(:,:,ia)))));
switch inner_meth
    case {'fgm2','ogm2'}
        cum_num = zeros(np,1);
        u0 = u;
        tt = 1;
        if strcmp(inner_meth,'fgm2')
            cc = 1;
        elseif strcmp(inner_meth,'ogm2')
            cc = 2;
        end
    case {'lalm','rlalm'}
        g = grad;
        h = denom0.*u-grad;
        if strcmp(inner_meth,'lalm')
            alpha = 1;
            rho = @(k) pi/k*sqrt(1-(pi/(2*k))^2)*(k>1)+(k==1)';
        elseif strcmp(inner_meth,'rlalm')
            alpha = 2;
            rho = @(k) pi/(2*k)*sqrt(1-(pi/(2*(2*k)))^2)*(k>1)+(k==1)';
        end
    otherwise
        error inner_meth;
end

info0 = arg.userfun(x,arg.userarg{:});
info = zeros(niter,1);

for iter = 1:niter
    tic;
    
    fprintf(['iter ' num2str(iter) ': ' count],0);
    for iset = 1:nblock
        switch inner_meth
            case {'fgm2','ogm2'}
                num = gain*(grad+R.cgrad(R,u))+(u-v);
                cum_num = cum_num+cc*tt*num;
                den = gain*(denom+R.denom(R,u))+1;
                z1 = proj(u-num./den);
                z2 = proj(u0-cum_num./den);
                tt = (1+sqrt(1+4*tt^2))/2;
                u = (1-1/tt)*z1+1/tt*z2;
                ib = ios(iset);
                ia = ib:nblock:na;
                grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*u-col(yi(:,:,ia)))));
            case {'lalm','rlalm'}
                num = gain*(rho(ii)*denom0.*u+(1-rho(ii))*g-rho(ii)*h+R.cgrad(R,u))+(u-v);
                den = gain*(rho(ii)*denom+R.denom(R,u))+1;
                u = proj(u-num./den);
                ib = ios(iset);
                ia = ib:nblock:na;
                grad = nblock*(Ab{ib}'*(col(wi(:,:,ia)).*(Ab{ib}*u-col(yi(:,:,ia)))));
                g = (rho(ii)*(alpha*grad+(1-alpha)*g)+g)/(rho(ii)+1);
                h = alpha*(denom0.*x-grad)+(1-alpha)*h;
        end
        switch outer_meth
            case 'nes83'
                xcur = u;
            case 'nes88'
                xcur = (1-theta(ip))*x+theta(ip)*u;
        end
        fprintf([back count],iset);
        
        if mod(ii,period(ip))==0
            xold = x;
            x = xcur;
            ip = ip+1;
            switch outer_meth
                case 'nes83'
                    v = x+theta(ip)*(1-theta(ip-1))/theta(ip-1)*(x-xold);
                    gain = t;
                case 'nes88'
                    v = u;
                    gain = t/theta(ip);
            end
            switch inner_meth
                case {'fgm2','ogm2'}
                    cum_num = zeros(np,1);
                    u0 = u;
                    tt = 1;
                case {'lalm','rlalm'}
                    g = grad;
                    h = denom0.*u-grad;
            end
            ii = 1;
        else
            ii = ii+1;
        end
    end
    
    iter_time = toc;
    info(iter) = arg.userfun(xcur,arg.userarg{:});
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
