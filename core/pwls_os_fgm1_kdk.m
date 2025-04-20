  function [xs, info] = pwls_os_fgm1_kdk(x, Ab, yi, R, varargin)
%|function [xs, info] = pwls_os_fgm1_kdk(x, Ab, yi, R, [options])
%|
%| penalized weighted least squares estimation / image reconstruction
%| (no onstraint) using fast gradient method with ordered subsets.
%|
%| cost(x) = (y-Ax)' W (y-Ax) / 2 + R(x)
%|
%| in
%|    x        [np 1]         initial estimate
%|    Ab       [nd np]        Gblock object, aij >= 0 required!
%|                            or sparse matrix (implies nsubset=1)
%|    yi       [nb na]        measurements (noisy sinogram data)
%|    R                       penalty object
%|
%| option
%|    niter                   # of iterations (default: 5)
%|    wi       [nb na]        weighting sinogram (default: [] for uniform)
%|    relax0   [1] or [2]     relax0 or (relax0, relax_rate)
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    chat
%|    type_mom                type of momentum (default: 'std')
%|                            'std': nesterov's standard momentum
%|                            'opt': kim's optimal momentum
%|    kdk_arg                 kdk-majorization parameter {g,d} (default: {[],[]})
%|                            g: 2d designed (high-pass) kernel
%|                               [] for standard sqs majorization (no power iter)
%|                            d0: designed diagonal weight [np 1]
%|                               [] for using d0 = diag{|G|'H|G|1}
%|
%| out
%|    xs       [np niter]     iterates
%|    info     [niter 1]      time
%|

if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.niter = 5;
arg.isave = [];
arg.userfun = @userfun_default;
arg.userarg = {};
arg.chat = false;
arg.wi = [];
arg.relax0 = 1;
arg.type_mom = 'std';
arg.kdk_arg = {[],[]};
arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);

Ab = block_op(Ab, 'ensure'); % make it a block object (if not already)
nblock = block_op(Ab, 'n');
starts = subset_start(nblock);

tov = @(x) masker(x,R.mask);
tom = @(x) embed(x,R.mask);
np = length(x);
na = size(yi,2);
one = ones(np,1,'single');

cpu etic

if ndims(x)~=2 || size(x,2)~=1
    fail 'bad x size';
end

wi = arg.wi;
if isempty(wi)
    wi = ones(size(yi));
end

% check input sinogram sizes for OS
if (ndims(yi) ~= 2) || (size(yi,2) == 1 && nblock > 1)
    fail 'bad yi size'
end
if (ndims(wi) ~= 2) || (size(wi,2) == 1 && nblock > 1)
    fail 'bad wi size'
end

relax0 = arg.relax0(1);
if length(arg.relax0) == 1
    relax_rate = 0;
elseif length(arg.relax0) == 2
    relax_rate = arg.relax0(2);
else
    error relax
end

if strcmp(arg.type_mom,'std')
    opt_mom = false;
elseif strcmp(arg.type_mom,'opt')
    opt_mom = true;
else
    fail 'bad momentum type';
end

g = arg.kdk_arg{1};
d0 = arg.kdk_arg{2};
if isempty(g)
    % standard sqs majorization
    std_sqs = true;
    d0 = Ab'*col(reshape(sum(Ab'),size(wi)).*wi)+R.denom(R,one);
    d0(d0==0) = inf;
else
    std_sqs = false;
    G = Gblur(R.mask,'type','conv,same','psf',g);
    C = abs(R.C1);
    lam = zeros(numel(R.mask),R.M);
    for l = 1:R.M, lam(:,l) = R.wt.col(l); end
    lam = lam(:);
    H = @(x) Ab'*(wi(:).*(Ab*x))+C'*(lam.*(C*x));
    if isempty(d0)
        d0 = abs(G)'*tom(H(tov(abs(G)*one))); 
        d0(d0==0) = inf;
    end
    % % approximate power iter
    %dd = H(one);
    %f = @(b) tov((G'*tom(dd.*(tov(G*(tom(b)./sqrt(d0))))))./sqrt(d0));
    %fprintf('compute approximate power iteration. ');
    %b = randn(np,1);
    %for iter = 1:1000
    %    fb = f(b);
    %    c = (b'*fb)/(b'*b);
    %    b = fb/norm(fb);
    %end
    % os-power
    Hb = @(x,ib) nblock*Ab{ib}'*(col(wi(:,ib:nblock:na)).*(Ab{ib}*x))+C'*(lam.*(C*x));
    f = @(b,ib) tov((G'*tom(Hb(tov(G*(tom(b)./sqrt(d0))),ib)))./sqrt(d0));
    fprintf('compute max eigenvalue using stoc power iter.\n');
    b = randn(np,1);
    for iter = 1:ceil(100/nblock)
        for iset = 1:nblock
            fb = f(b,starts(iset));
            c = (b'*fb)/(b'*b);
            b = fb/norm(fb);
        end
    end
    % % vr-pca
    %b = randn(np,1);
    %eta = 1/sqrt(na);
    %Ai = Gblock(Ab,na);
    %[count,back] = loop_count_str(na);
    %fprintf('find top eigenvector using vr-pca.\n');
    %for iter = 1:2
    %    u = tov((G'*tom(H(tov(G*(tom(b)./sqrt(d0))))))./sqrt(d0));
    %    c = (b'*u)/(b'*b);
    %    fprintf('c = %g\n',c);
    %    bt = b;
    %    view = randperm(na);
    %    fprintf(['iter %g: ' count],iter,0);
    %    for ia = 1:na
    %        s1 = tov(G*tom(bt-b)./sqrt(d0));
    %        s2 = na*Ai{view(ia)}'*(wi(:,view(ia)).*(Ai{view(ia)}*s1))+C'*(lam.*(C*s1));
    %        s3 = tov((G'*tom(s2))./sqrt(d0));
    %        bb = bt+eta*(s3+u);
    %        bt = bb/norm(bb);
    %        fprintf([back count],ia);
    %    end
    %    fprintf('\n');
    %    b = bt;
    %end
    %fprintf('compute rayleigh quotient: ');
    %u = tov((G'*tom(H(tov(G*(tom(b)./sqrt(d0))))))./sqrt(d0));
    %c = (b'*u)/(b'*b);
    fprintf('c = %g\n',c);
end

xs = zeros(np, length(arg.isave));
if any(arg.isave == 0)
    xs(:, arg.isave == 0) = x;
end

%info = zeros(niter,?); % do not initialize since size may change

% initilization
zold = x;
told = 1;

% iterate
for iter = 1:arg.niter
    ticker(mfilename, iter, arg.niter)

    relax = relax0 / (1 + relax_rate * (iter-1));

    % loop over subsets
    for iset = 1:nblock
        ib = starts(iset);
        ia = ib:nblock:na;

        num = nblock*Ab{ib}'*(col(wi(:,ia)).*(Ab{ib}*x-col(yi(:,ia))))+R.cgrad(R,x);
        if std_sqs
            dec = num./d0;
        else
            dec = tov(G*((G'*tom(num))./d0))/c;
        end
        z = x-relax*dec;
        t = (1+sqrt(1+4*told^2))/2;
        if opt_mom
            x = z+(told-1)/t*(z-zold)+told/t*(z-x);
        else
            x = z+(told-1)/t*(z-zold);
        end
        zold = z;
        told = t;
    end

    if any(arg.isave == iter)
        xs(:, arg.isave == iter) = x;
    end
    info(iter,:) = arg.userfun(x, arg.userarg{:});
end


% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%    x = evalin('caller', 'x');
    printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');
