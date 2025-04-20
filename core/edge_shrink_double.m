  function x = edge_shrink_double(y, R, den, varargin)
%|function x = edge_shrink_double(y, R, den, [option])
%|
%| compute shrink operation of a proximal average approximation of R
%| (in double precision)
%|
%| in
%|    y        [np 1]         reference image
%|    R                       penalty object (see Reg1.m)
%|    den      [np 1]         denominator/weight
%|
%| option
%|    voxmax   [1  2]         [min max] (default: [-inf inf])
%|    use_abs                 use abs potential func (default: false)
%|
%| out
%|    x        [np 1]         shrinked image
%|

if nargin<3, help(mfilename), error(mfilename), end

% defaults
arg.voxmax = [-inf inf];
arg.use_abs = false;
arg = vararg_pair(arg,varargin);

% regularizer parameters
mask = R.mask;                 % image mask
dim = R.dim;                   % image dimension
M = R.M;                       % number of offsets
pot = R.pot_type;              % potential function name
phi = R.pot;                   % potential functions
kappa = col(R.cdp_arg{1});     % voxel-dependent kappa
beta = R.beta;                 % regularization parameters
offsets = R.offsets;           % finite-difference offsets
delta = R.pot_params;          % corner-rounding parameter
nthread = R.nthread;           % number of threads

% initialization
tov = @(x) x(mask);
tom = @(x) embed(x,mask);
y = col(tom(y));
den = col(tom(den));
is_box = arg.voxmax(1)~=-inf || arg.voxmax(2)~=inf;
proj = @(x) min(max(x,arg.voxmax(1)),arg.voxmax(2));
N = 2*M+is_box;
winv = div0(1,den);

% shrink operation
if is_box, x = proj(y)/N; else x = zeros(size(y)); end
coef = parallel_implicit_shrink_double_mex(y,winv,kappa,offsets,beta,pot,delta,N,nthread);
x = x+(2*M*y+coef.*winv)/N;
x = tov(reshape(x,dim));

function coef = implicit_shrink_mat(y,winv,kappa,offsets,beta,phi,delta,N)
coef = zeros(size(y));
for m = 1:length(offsets)
    cy = y(1:end-offsets(m))-y(1+offsets(m):end);
    cwc = winv(1:end-offsets(m))+winv(1+offsets(m):end);
    kap = kappa(1:end-offsets(m)).*kappa(1+offsets(m):end);
    if 0
        cy_shrink = abs_shrink(cy,N*beta(m)*kap.*cwc,delta);
    else
        cy_shrink = phi{m}.meth.shrink(phi{m},cy,N*beta(m)*kap.*cwc);
    end
    alpha = [zeros(offsets(m),1); div0(cy_shrink-cy,cwc); zeros(offsets(m),1)];
    coef = coef-alpha(1:end-offsets(m))+alpha(1+offsets(m):end);
end


