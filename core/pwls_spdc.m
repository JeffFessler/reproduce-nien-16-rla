  function [xs, info] = pwls_spdc(x, Ab, yi, R, varargin)
%|function [xs, info] = pwls_spdc(x, Ab, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| using the stochastic primal-dual coordinate algorithm
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|    x        [np 1]         initial estimate
%|    Ab       [nd np]        Gblock object, aij >= 0 required!
%|                            or sparse matrix (implies nsubset=1)
%|    yi       [nb na]        measurements (noisy sinogram data)
%|    R                       penalty object (see Reg1.m), can be []
%|
%| option
%|    nepoch                  # of epochs (default: 1)
%|    isave                   save images (default: nepoch)
%|    wi       [nb na]        weighting sinogram (default: [] for uniform)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default [0 inf])
%|    aai      [nb na]        precomputed row sums of |Ab|
%|    pk                      weighted sampling (default: [])
%|    wt_proj                 weighted projection (default: 1)
%|    rho                     al penalty parameter of sinogram (default: 1)
%|    eta                     al penalty parameter of difference image (default: 1)
%|    theta                   relaxation parameter (default: 1)
%|    delta                   strongly convex approximate term (default: 0)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat
%|
%| out
%|    xs       [np niter]     iterates
%|    info     [niter 1]      time
%|

if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.nepoch = 1;
arg.isave = [];
arg.userfun = @userfun_default;
arg.userarg = {};
arg.pixmax = inf;
arg.chat = false;
arg.wi = [];
arg.aai = [];
arg.pk = [];
arg.wt_proj = 1;
arg.rho = 1;
arg.eta = 1;
arg.theta = 1;
arg.delta = 0;
arg = vararg_pair(arg,varargin);

nepoch = arg.nepoch;
isave = arg.isave;
if isempty(isave)
    isave = nepoch;
end

Ab = block_op(Ab,'ensure'); % make it a block object (if not already)
nblock = block_op(Ab,'n');

cpu etic

wi = arg.wi;
if isempty(wi)
    wi = ones(size(yi));
end
swi = sqrt(wi);

if isempty(arg.aai)
    % a_i = sum_j |a_ij|, requires real a_ij and a_ij >= 0
    arg.aai = reshape(sum(Ab'),size(yi));
end

% check input sinogram sizes for OS
if (ndims(yi)~=2) || (size(yi,2)==1 && nblock>1)
    fail 'bad yi size';
end
if (ndims(wi)~=2) || (size(wi,2)==1 && nblock>1)
    fail 'bad wi size';
end

if length(arg.pixmax)==2
    pixmin = arg.pixmax(1);
    pixmax = arg.pixmax(2);
elseif length(arg.pixmax)==1
    pixmin = 0;
    pixmax = arg.pixmax;
else
    fail 'bad pixmax';
end

% likelihood denom, if not provided
denom = Ab'*col(arg.aai.*wi); % requires real a_ij and a_ij >= 0
denom(denom==0) = inf;

rho = arg.rho;
eta = arg.eta;
theta = arg.theta;
delta = arg.delta;

[nb,na] = size(yi);
nd = nb*na;

x = x(:);
np = length(x);
xs = zeros(np,length(isave));
if any(isave==0)
    xs = [xs x];
end

M = R.M;
C1 = R.C1;
nr = C1.size(1);
pot = R.pot{1};

tov = @(x) x(R.mask);
tom = @(x) embed(x,R.mask);

n = nd+nr;
nbatch = nblock+M;
pk = arg.pk;
if isempty(pk)
    pk = ones(1,nbatch)/nbatch;
end
if length(pk)~=nbatch
    fail 'bad weighted sampling';
end
Bk = zeros(1,nbatch);
z = cell(1,nbatch);
for ibatch = 1:nbatch
    if ibatch<=nblock  % loss part
        Bk(ibatch) = nb*length(ibatch:nblock:na);
    else  % regularization part
        Bk(ibatch) = nr/M;
    end
    z{ibatch} = zeros(Bk(ibatch),1);
end

% initialization
xb = x;
u = zeros(np,1);
block = 0;
epoch = 1;
info0 = arg.userfun(x);
info = zeros(nepoch,1);
den = rho*denom+eta*4*M;

[count,back] = loop_count_str(nblock);
fprintf('solve 2d x-ray ct image recon using spdc...\n');
fprintf(['epoch ' num2str(epoch) ': ' count],block);
% iterate
while epoch<=nepoch
    % z-update
    ibatch = randp(pk);
    if ibatch<=nblock
        iblock = ibatch;
        ia = iblock:nblock:na;
        beta = pk(ibatch)/Bk(ibatch)/rho;
        zb = z{ibatch}+(col(swi(:,ia)).*(Ab{iblock}*xb))/beta;
        beta1 = beta+delta;
        zb1 = beta/beta1*zb;
        znew = zb1-((beta1*n*(col(swi(:,ia).*yi(:,ia)))+beta1*zb1)/(beta1*n+1))/beta1;
        du = Ab{iblock}'*(col(swi(:,ia)).*(znew-z{ibatch}))/n;
        block = block+1;
        fprintf([back count],block);
    else
        ireg = ibatch-nblock;
        beta = pk(ibatch)/Bk(ibatch)/eta;
        zb = z{ibatch}+col(C1.Cc{ireg}*tom(xb))/beta;
        beta1 = beta+delta;
        zb1 = beta/beta1*zb;
        znew = zb1-(pot.shrink(beta1*zb1,n*R.wt.col(ireg)*beta1))/beta1;
        du = tov(C1.Cc{ireg}'*reshape(znew-z{ibatch},size(R.mask))/n);
    end
    z{ibatch} = znew;
    num = u+du/pk(ibatch);
    % im(embed(num,R.mask)); cbar; pause;
    % u-update
    u = u+du;
    % im(embed(u,R.mask)); cbar; pause;
    % x-update
    xold = x;
    x = den./(den+delta).*(x-num./den);
    x = max(x,pixmin);
    x = min(x,pixmax);
    xb = x+theta*(x-xold);
    
    if block==nblock
        info(epoch) = arg.userfun(x);
        fprintf([' info = ' num2str(info(epoch)) '\n']);
        if any(isave==epoch)
            xs = [xs x];
        end
        epoch = epoch+1;
        block = 0;
        if epoch<=nepoch
            fprintf(['epoch ' num2str(epoch) ': ' count],block);
        end
    end
end
info = [info0; info];


% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
