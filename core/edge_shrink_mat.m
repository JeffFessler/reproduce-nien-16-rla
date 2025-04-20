  function x = edge_shrink_mat(z, R, eta, den, varargin)
%|function x = edge_shrink_mat(z, R, eta, den, [option])
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
%|    use_abs                 use abs potential func (default: false)
%|
%| out
%|    x        [np 1]         shrinked image
%|

if nargin<4, help(mfilename), error(mfilename), end

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

% kappa = double(kappa);
% z = double(z);
% eta = double(eta);
% den = double(den);

z = col(tom(z));
den = col(tom(den));
is_box = arg.voxmax(1)~=-inf || arg.voxmax(2)~=inf;
proj = @(x) min(max(x,arg.voxmax(1)),arg.voxmax(2));
N = 2*M+is_box;
dinv = div0(1,den);

% shrink operation
if is_box, x = proj(z)/N; else x = zeros(size(z)); end
coef = implicit_shrink_mat(z,eta,dinv,kappa,offsets,beta,phi,delta,N);
x = x+(2*M*z+coef.*dinv)/N;
x = tov(reshape(x,dim));

function out = abs_shrink(z, reg, delta)
out = sign(z).*max(abs(z)-delta*reg,0);

function coef = implicit_shrink_mat(z,eta,dinv,kappa,offsets,beta,phi,delta,N)
coef = zeros(size(z));
% zero boundary condition
for m = 1:length(offsets)
    cz = z(1:end-offsets(m))-z(1+offsets(m):end);
    cwc = dinv(1:end-offsets(m))+dinv(1+offsets(m):end);
    kap = kappa(1:end-offsets(m)).*kappa(1+offsets(m):end);
    if 0
        cz_shrink = abs_shrink(cz,eta*N*beta(m)*kap.*cwc,delta);
    else
        cz_shrink = phi{m}.meth.shrink(phi{m},cz,eta*N*beta(m)*kap.*cwc);
    end
    alpha = [zeros(offsets(m),1); div0(cz_shrink-cz,cwc); zeros(offsets(m),1)];
    coef = coef-alpha(1:end-offsets(m))+alpha(1+offsets(m):end);
    
end

% % circulant boundary condition
% for m = 1:length(offsets)
%     cz  = z-circshift(z,offsets(m));
%     cwc = dinv+circshift(dinv,offsets(m));
%     kap = kappa.*circshift(kappa,offsets(m));
%     if 0
%         cz_shrink = abs_shrink(cz,eta*N*beta(m)*kap.*cwc,delta);
%     else
%         cz_shrink = double(phi{m}.meth.shrink(phi{m},cz,eta*N*beta(m)*kap.*cwc));
%     end
%     alpha = div0(cz_shrink-cz,cwc);
%     coef = coef+alpha-circshift(alpha,-offsets(m));
% end