  function x = edge_shrink1(z, R, eta, den, varargin)
%|function x = edge_shrink1(z, R, eta, den, [option])
%|
%| compute shrink operation of a proximal average approximation of R
%|
%| in
%|    z        [np 1]         reference image
%|    R                       penalty object (see Reg1.m)
%|    eta                     relaxed step size
%|    den      [np 1]         denominator/weight
%|
%| option
%|    voxmax   [1  2]         [min max] (default: [-inf inf])
%|    K                       overwritten block size by user (default: [])
%|
%| out
%|    x        [np 1]         shrinked image
%|

if nargin<4, help(mfilename), error(mfilename), end

% defaults
arg.voxmax = [-inf inf];
arg.K = [];
arg = vararg_pair(arg,varargin);

% regularizer parameters
mask = R.mask;                 % image mask
dim = R.dim;                   % image dimension
M = R.M;                       % number of offsets
pot = R.pot_type;              % potential function name
phi = R.pot;                   % potential function object
kappa = col(R.cdp_arg{1});     % voxel-dependent kappa
beta = R.beta;                 % regularization parameters
offsets = R.offsets;           % finite-difference offsets
delta = R.pot_params;          % corner-rounding parameter
nthread = R.nthread;           % number of threads

% initialization
tov = @(x) x(mask);
tom = @(x) embed(x,mask);
z = col(tom(z));
den = col(tom(den));
is_box = arg.voxmax(1)~=-inf || arg.voxmax(2)~=inf;
proj = @(x) min(max(x,arg.voxmax(1)),arg.voxmax(2));
dinv = div0(1,den);
if isempty(arg.K)
    K = 2*M+is_box;
else
    K = arg.K;
end
res = K-is_box;

% shrink operation
if is_box, x = proj(z)/K; else x = zeros(size(z)); end
coef = parallel_implicit_shrink1_mex(z,eta,dinv,kappa,offsets,beta,pot,delta,K,nthread);
% coef = implicit_shrink1_mat(z,eta,dinv,kappa,offsets,beta,phi,K);
x = x+(res*z+coef.*dinv)/K;
x = tov(reshape(x,dim));

function coef = implicit_shrink1_mat(z,eta,dinv,kappa,offsets,beta,phi,K)
coef = zeros(size(z));
for m = 1:length(offsets)
    cz = z(1:end-offsets(m))-z(1+offsets(m):end);
    cwc = dinv(1:end-offsets(m))+dinv(1+offsets(m):end);
    kap = kappa(1:end-offsets(m)).*kappa(1+offsets(m):end);
    cz_shrink = phi{m}.meth.shrink(phi{m},cz,eta*K*beta(m)*kap.*cwc);
    alpha = [zeros(offsets(m),1); div0(cz_shrink-cz,cwc); zeros(offsets(m),1)];
    coef = coef-alpha(1:end-offsets(m))+alpha(1+offsets(m):end);
end