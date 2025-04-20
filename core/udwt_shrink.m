  function x = udwt_shrink(y, nl, beta, den, varargin)
%|function x = udwt_shrink(y, nl, beta, den, [option])
%|
%| compute shrink operation of a proximal average approximation of sparse
%| undecimated haar wavelet transform
%|
%| in
%|    y        [nx ny]        reference image
%|    nl                      # level of undecimated wavelet transform
%|    beta     [nl  3]        shrinkage param for each level and direction
%|    den      [nx ny]        denominator/weight
%|
%| option
%|    scalar                  factor for scaling haar wavelet filters (default: 1/sqrt(2))
%|    voxmax   [1  2]         [min max] (default: [-inf inf])
%|
%| out
%|    x        [nx ny]        shrinked image
%|

if nargin<4, help(mfilename), error(mfilename), end

% defaults
arg.scalar = 1/sqrt(2);
arg.voxmax = [-inf inf];
arg = vararg_pair(arg,varargin);

% initialization
is_box = arg.voxmax(1)~=-inf || arg.voxmax(2)~=inf;
proj = @(x) min(max(x,arg.voxmax(1)),arg.voxmax(2));
N = sum(4.^(1:nl))*3+is_box; % ignore last approx coef
winv = div0(1,den);
const = arg.scalar/sqrt(2);
lo0 = const*[1  1];
hi0 = const*[1 -1];

% shrink operation
if is_box, x = proj(y)/N; else x = zeros(size(y)); end
for l = 1:nl
    lo = col(repmat(lo0,[2^(l-1) 1]))*const^(l-1);  % column vector
    hi = col(repmat(hi0,[2^(l-1) 1]))*const^(l-1);  % column vector
    % dim 1 (x-axis)
    x = x+(row_update(y,hi,lo',winv,N*beta(l,1)))/N;
    % dim 2 (y-axis)
    x = x+(row_update(y,lo,hi',winv,N*beta(l,2)))/N;
    % dim 1 & 2
    x = x+(row_update(y,hi,hi',winv,N*beta(l,3)))/N;
end

function inc = row_update(y,d1,d2,winv,beta)
% ker = d1*d2 (d1: column vector, d2: row vector)
% note 1: conv2(conv2(x,d1),d2)==conv2(x,d1*d2)
% note 2: rot90(ker,2) = d1(end:-1:1)*d2(end:-1:1)'
cwc = conv2(conv2(winv,d1(end:-1:1).^2,'valid'),d2(end:-1:1).^2,'valid');
cy  = conv2(conv2(y,d1(end:-1:1),'valid'),d2(end:-1:1),'valid');
fx = @(z) sign(z).*max(abs(z)-beta*cwc,0);
% fx = @(z) sign(z).*(abs(z)/2-beta*cwc/2+sqrt((abs(z)/2+beta*cwc/2).^2-(beta*cwc).^2)).*(abs(z)>beta*cwc);
alpha = div0(fx(cy)-cy,cwc);

coef = conv2(conv2(alpha,d1),d2);
inc = numel(d1)*numel(d2)*y+coef.*winv;
