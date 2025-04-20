  function x = udwt_shrink0(y, L, beta, den, varargin)
%|function x = udwt_shrink0(y, L, beta, den, [option])
%|
%| compute shrink operation of a proximal average approximation of sparse
%| undecimated haar wavelet transform
%|
%| in
%|    y        [nx ny]        reference image
%|    L                       # level of undecimated wavelet transform
%|    beta     [L   3]        shrinkage param for each level and direction
%|    den      [nx ny]        denominator/weight
%|
%| option
%|    voxmax   [1  2]         [min max] (default: [-inf inf])
%|
%| out
%|    x        [nx ny]        shrinked image
%|

if nargin<4, help(mfilename), error(mfilename), end

% defaults
arg.voxmax = [-inf inf];
arg = vararg_pair(arg,varargin);

% initialization
is_box = arg.voxmax(1)~=-inf || arg.voxmax(2)~=inf;
proj = @(x) min(max(x,arg.voxmax(1)),arg.voxmax(2));
N = sum(4.^(1:L))*3+is_box; % ignore last approx coef
winv = div0(1,den);
H = 1;

% shrink operation
if is_box, x = proj(y)/N; else x = zeros(size(y)); end
for l = 1:L
    h0 = [1/2; zeros(2^(l-1)-1,1); 1/2];
    h1 = [1/2; zeros(2^(l-1)-1,1); -1/2];
    % horizontal
    c = conv2(conv2(H,h1),h0');
    x = x+(row_update(y,c,winv,N*beta(l,1)))/N;
    % vertical
    c = conv2(conv2(H,h0),h1');
    x = x+(row_update(y,c,winv,N*beta(l,2)))/N;
    % diagonal
    c = conv2(conv2(H,h1),h1');
    x = x+(row_update(y,c,winv,N*beta(l,3)))/N;
    % update H
    H = conv2(conv2(H,h0),h0');
end

function inc = row_update(y,c,winv,beta)
cf = rot90(c,2);
cwc = conv2(winv,cf.^2,'valid');
cy = conv2(y,cf,'valid');
fx = @(z) sign(z).*max(abs(z)-beta*cwc,0);
% fx = @(z) sign(z).*(abs(z)/2-beta*cwc/2+sqrt((abs(z)/2+beta*cwc/2).^2-(beta*cwc).^2)).*(abs(z)>beta*cwc);
alpha = div0(fx(cy)-cy,cwc);

coef = conv2(alpha,c);
inc = y*numel(c)+coef.*winv;

% [nx,ny] = size(y);
% [sx,sy] = size(c);
% inc = zeros(nx,ny);
% for ix = 1:sx
%     for iy = 1:sy
%         alp = alpha(ix:sx:end,iy:sy:end);
%         [bx,by] = size(alp);
%         alp0 = padarray(padarray(alp(ceil(1/sx:1/sx:end),ceil(1/sy:1/sy:end)),[ix-1 iy-1],0,'pre'),[nx-bx*sx-(ix-1) ny-by*sy-(iy-1)],0,'post');
%         wc = winv.*padarray(padarray(repmat(c,[bx by]),[ix-1 iy-1],0,'pre'),[nx-bx*sx-(ix-1) ny-by*sy-(iy-1)],0,'post');
%         inc = inc+(y+alp0.*wc);
%     end
% end
