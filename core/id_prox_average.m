  function [x, info] = id_prox_average(x, y, R, varargin)
%|function [x, info] = id_prox_average(x, y, R, [options])
%|
%| image denoise using the fast (nesterov 1988) dual proximal gradient method
%| with memory-efficient proximal average updates.
%|
%| in
%|    x        [nx ny]        initial estimation
%|    y        [nx ny]        noisy blury image
%|    R                       penalty object (see Reg1.m)
%|
%| option
%|    niter                   # of iterations (default: 1)
%|    pixmax   [1] or [2]     max pixel value, or [min max] (default: [0 inf])
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat                    
%|
%| out
%|    x        [nx ny]        reconstructed image
%|    info     [niter 1]      output info
%|

if nargin < 3, help(mfilename), error(mfilename), end

% defaults
arg.niter = 1;
arg.pixmax = inf;
arg.userfun = @userfun_default;
arg.userarg = {};
arg.chat = false;
arg = vararg_pair(arg, varargin);

% parameters
tov = @(x) x(R.mask);
tom = @(x) embed(x,R.mask);
np = sum(R.mask(:));
one = ones(np,1,'single');
niter = arg.niter;
cpu etic;

% box constraint
if length(arg.pixmax)==2
    pixmin = arg.pixmax(1);
    pixmax = arg.pixmax(2);
elseif length(arg.pixmax)==1
    pixmin = 0;
    pixmax = arg.pixmax;
else
    fail 'bad pixmax'
end
box = [pixmin pixmax];

% initilization
x = max(min(x,pixmax),pixmin);
z = zeros(size(x));
v = z;
info = zeros(niter,1);

[count,back] = loop_count_str(niter);

% image reconstruction
fprintf(['image recon: fast dual proximal gradient ' count],0);
for iter = 1:niter
    t = 2/(iter+1);
    % t = 2/(iter+1)^0.8;
    % u-update
    u = (1-t)*z+t*v;
    % v-update
    w = v-(u-y)/t;
    den = one/t;
    v = w-1/t*tom(edge_shrink(tov(t*w),R,den,'voxmax',box));  % single precision (unstable around precision)
    % v = w-1/t*tom(edge_shrink_double(double(tov(t*w)),R,double(den),'voxmax',box));
    % z-update
    z = (1-t)*z+t*v;
    % x-update
    x = y-z;
    
    % update info
    info(iter) = arg.userfun(x,arg.userarg{:});
    fprintf([back count],iter);
end
fprintf('\n');

% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
