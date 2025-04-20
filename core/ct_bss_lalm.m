  function [xrec, info] = ct_bss_lalm(x0, A, yi, R, varargin)
%|function [xrec, info] = ct_bss_lalm(x0, A, yi, R, [options])
%|
%| model-based 3d x-ray ct image reconstruction using simulated linearized
%| augmented lagrangian method with block separable surrogate (ordered-
%| subsets inside)
%|
%| cost(x) = (y-Ax)'W(y-Ax)/2+R(x)
%|
%| in
%|    x0       [nx ny nz]     initial estimate
%|    A        [nd np]        Gcone object
%|    yi       [ns nt na]     measurements (noisy sinogram data)
%|    R                       penalty object (see Reg1.m), can be []
%|
%| option
%|    nslab                   # of slabs (default: 5)
%|    bss_type                bss options (default: 'uniform')
%|    grad_type               gradient options (default: 'full')
%|    ncomm                   # of communications (default: 1)
%|    nblock                  # of ordered-subsets (default: 12)
%|    niter                   # of os iterations (default: 5)
%|    nditer                  # of denoising iterations (default: 50)
%|    isave                   indices of images to be saved (default: [])
%|    path                    path to saved images (default: './')
%|    wi       [ns nt na]     weighting sinogram (default: [] for uniform)
%|    voxmax   [1] or [2]     max voxel value, or [min max] (default: [0 inf])
%|    userfun  @              user defined function handle (see default below)
%|                            taking arguments (x, userarg{:})
%|    userarg  {}             user arguments to userfun (default {})
%|    rho                     al penalty parameter for u = Ax (default: 1)
%|    eta                     al penalty parameter for v = Sx (default: 1)
%|
%| out
%|    xrec     [nx ny nz]     reconstructed image
%|    info     [niter 1]      outcome of user defined function
%|

if nargin<4, help(mfilename), error(mfilename), end

% defaults
arg.nslab = 5;
arg.bss_type = 'uniform';
arg.grad_type = 'full';
arg.ncomm = 1;
arg.nblock = 12;
arg.niter = 5;
arg.nditer = 50;
arg.isave = [];
arg.path = './';
arg.userfun = @userfun_default;
arg.userarg = {};
arg.voxmax = inf;
arg.wi = [];
arg.rho = 1;
arg.eta = 1;
arg = vararg_pair(arg,varargin);

nslab = arg.nslab;
ncomm = arg.ncomm;
nblock = arg.nblock;
niter = arg.niter;
nditer = arg.nditer;

Ab0 = Gblock(A,nblock);

if ~exist(arg.path,'dir')
  mkdir(arg.path);
end
log = [ ...
    'simulate x-ray ct image reconstruction using bss-lalm...\n' ...
    '==========================================================\n' ...
    'parameter list:\n' ...
    '# of slabs: ' num2str(nslab) '\n' ...
    'type of bss: ' arg.bss_type '\n' ...
    'type of grad: ' arg.grad_type '\n' ...
    '# of sino communications: ' num2str(ncomm) '\n' ...
    '# of subsets used in os: ' num2str(nblock) '\n' ...
    '# of forw/back-proj per image update: ' num2str(niter) '\n' ...
    '# of denoise per end-slice update: ' num2str(nditer) '\n' ...
    '==========================================================\n' ...
    ];

DEBUG = 0;

% bss options
if ~strcmp(arg.bss_type,'uniform') && ~strcmp(arg.bss_type,'non-uniform')
    fail 'bad bss option';
end

% grad options
if strcmp(arg.grad_type,'full')
    ;
elseif strcmp(arg.grad_type,'partial')
    iset = nblock;
    next = @(i) mod(i,nblock)+1;
    bit_reverse = subset_start(nblock);
else
    fail 'bad grad option';
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
    fail 'illegal voxmax';
end

% al penalty parameter rho
if isnumeric(arg.rho)
    if arg.rho>0
        rho = str2func(['@(k)' num2str(arg.rho)]);
    else
        fail 'illegal al penalty parameter rho?!';
    end
else
    rho = str2func(arg.rho);
end

% al penalty parameter eta
if isnumeric(arg.eta)
    if arg.eta>0
        eta = str2func(['@(k)' num2str(arg.eta)]);
    else
        fail 'illegal al penalty parameter eta?!';
    end
else
    eta = str2func(arg.eta);
end

if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],x0);
end

% progress status
[count1,back1] = loop_count_str(niter);
[count2,back2] = loop_count_str(nditer);

% matrix-vector conversion
tov = @(x) x(A.imask); % 3d (xyz) to 1d (xyz)
tom = @(x) embed(x,A.imask); % 1d (xyz) to 3d (xyz)

% setup bss and reg
bss = setup_bss(A,nslab,'wi',wi,'how',arg.bss_type);
if DEBUG
    y1 = A*tov(x0);
    y2 = 0;
    for slab = 1:nslab
        y2 = y2+bss.A{slab}*bss.tov{slab}(bss.exm{slab}(x0));
    end
    if norm(y1-y2)/sqrt(length(y1))>1, fail('bss forw. projection does NOT match non-bss forw. projection.'); end
    clear y1 y2;
end
clear A;
reg = setup_reg(R,bss.iz);
if DEBUG
    z1g = R.cgrad(R,tov(x0));
    z1d = R.denom(R,tov(x0));
    z2g = []; z2d = [];
    for slab = 1:nslab
        xt = [];
        if slab>1, xt = get_element(bss.tov{slab-1}(bss.exm{slab-1}(x0)),3,reg.c_map{slab-1}); end
        xc = bss.tov{slab}(bss.exm{slab}(x0));
        xb = [];
        if slab<nslab, xb = get_element(bss.tov{slab+1}(bss.exm{slab+1}(x0)),2,reg.c_map{slab+1}); end
        xp = [xt; xc; xb];
        z2g = [z2g; get_element(reg.R_tcb{slab}.cgrad(reg.R_tcb{slab},xp),1,reg.tcb_map{slab})];
        z2d = [z2d; get_element(reg.R_tcb{slab}.denom(reg.R_tcb{slab},xp),1,reg.tcb_map{slab})];
    end
    if norm(z1g-z2g)~=0, fail('bss reg. gradient does NOT match non-bss reg. gradient.'); end
    if norm(z1d-z2d)~=0, fail('bss reg. denominator does NOT match non-bss reg. denominator.'); end
    clear z1g z1d z2g z2d xt xc xb xp;
end
clear R;

% initialize lalm
fprintf('***** initialization *****\n');
x0 = tov(x0); x0 = max(x0,voxmin); x0 = min(x0,voxmax); x0 = tom(x0);
sAx = 0;
x = cell(nslab,1);
for slab = 1:nslab
    fprintf(['    @ slab ' num2str(slab) ':\n']);
    fprintf('        initialize x...\n');
    x{slab} = bss.tov{slab}(bss.exm{slab}(x0));
    fprintf('        compute Ax...\n');
    if strcmp(arg.grad_type,'full')
        sAx = sAx+bss.A{slab}*x{slab};
    else
        Ab = Gblock(bss.A{slab},nblock);
        sAx = sAx+Ab{bit_reverse(iset)}*x{slab};
    end
end
xcol = cell2mat(x);
fprintf('    --- communication (compute residual using MPI_AllReduce) ---\n');
if strcmp(arg.grad_type,'full')
    res = sAx-yi(:);
else
    ia = bit_reverse(iset):nblock:Ab.odim(end);
    res = sAx-col(yi(:,:,ia));
end
z = cell(1,nslab); g = cell(1,nslab);
for slab = 1:nslab
    fprintf(['    @ slab' num2str(slab) ':\n']);
    fprintf('        initialize zeta and g...\n');
    if strcmp(arg.grad_type,'full')
        z{slab} = bss.A{slab}'*(wi(:).*res);
    else
        Ab = Gblock(bss.A{slab},nblock);
        z{slab} = nblock*Ab{bit_reverse(iset)}'*(col(wi(:,:,ia)).*res);
    end
    g{slab} = rho(1)*z{slab};
end
if strcmp(arg.grad_type,'partial')
    iset = next(iset);
end
%figure; im(squeeze(wi(:,end/2,:))); cbar; pause;
fprintf('    --- communication (exchange end-slices with adjacent slabs) ---\n');
vt = cell(1,nslab); et = cell(1,nslab);
vb = cell(1,nslab); eb = cell(1,nslab);
xt = cell(1,nslab); xb = cell(1,nslab);
for slab = 1:nslab
    fprintf(['    @ slab ' num2str(slab) ':\n']);
    fprintf('        initialize vt and et...\n');
    vt{slab} = []; et{slab} = [];
    if slab>1
        vt{slab} = get_element(x{slab-1},3,reg.c_map{slab-1});
        et{slab} = 0*vt{slab};
        xb{slab-1} = vt{slab}+et{slab};
    end
    fprintf('        initialize vb and eb...\n');
    vb{slab} = []; eb{slab} = [];
    if slab<nslab
        vb{slab} = get_element(x{slab+1},2,reg.c_map{slab+1});
        eb{slab} = 0*vb{slab};
        xt{slab+1} = vb{slab}+eb{slab};
    end
end

info0 = arg.userfun(tov(x0),arg.userarg{:});
info = zeros(ncomm,1);
xrec = x0;
if any(arg.isave==0)
    fld_write([arg.path 'x_iter_0.fld' ],xrec);
end
log = strcat(log,sprintf('iter 0: info = %f\\n',info0));

% image recon using bss-lalm
for comm = 1:ncomm
    fprintf(['***** iter ' num2str(comm) ' *****\n']);
    sAx = 0;
    for slab = 1:nslab
        fprintf(['    @ slab ' num2str(slab) ':\n']);
        fprintf('        initialize os-mom...\n');
        xk = x{slab};
        sk = rho(comm)*z{slab}+(1-rho(comm))*g{slab};
        if slab>1
            dt = single(reg.c_map{slab}==2); dt = dt(reg.c_map{slab}~=0);
            ct = set_element(zeros(size(xk),'single'),xt{slab},2,reg.c_map{slab});
        end
        if slab<nslab
            db = single(reg.c_map{slab}==3); db = db(reg.c_map{slab}~=0);
            cb = set_element(zeros(size(xk),'single'),xb{slab},3,reg.c_map{slab});
        end
        Ab = Gblock(bss.A{slab},nblock);
        A1b = bss.A{slab}*ones(bss.A{slab}.np,1,'single');
        ws = wi.*reshape(bss.A1./A1b,size(wi));
        dwls = bss.A{slab}'*(ws(:).*A1b);
        clear A1b;
        na = bss.A{slab}.odim(end);
        cum_num = 0;
        t = 1;
        fprintf(['        update x... ' count1],0);
        for iter = 1:niter
            for iblock = subset_start(nblock)'
                xp = [vt{slab}; x{slab}; vb{slab}];
                ia = iblock:nblock:na;
                % numerator
                num = get_element(reg.R_tcb{slab}.cgrad(reg.R_tcb{slab},xp),1,reg.tcb_map{slab})+...
                    sk+rho(comm)*nblock*Ab{iblock}'*(col(ws(:,:,ia)).*(Ab{iblock}*(x{slab}-xk)));
                if slab>1
                    num = num+eta(comm)*(x{slab}.*dt-ct);
                end
                if slab<nslab
                    num = num+eta(comm)*(x{slab}.*db-cb);
                end
                cum_num = cum_num+t*num;
                % denominator
                den = get_element(reg.R_tcb{slab}.denom(reg.R_tcb{slab},xp),1,reg.tcb_map{slab})+rho(comm)*dwls;
                if slab>1
                    den = den+eta(comm)*dt;
                end
                if slab<nslab
                    den = den+eta(comm)*db;
                end
                % image update
                z1 = x{slab}-num./den;
                z1 = max(z1,voxmin);
                z1 = min(z1,voxmax);

                z2 = xk-cum_num./den;
                z2 = max(z2,voxmin);
                z2 = min(z2,voxmax);

                t = (1+sqrt(1+4*t^2))/2;
                x{slab} = (1-1/t)*z1+1/t*z2;
            end
            fprintf([back1 count1],iter);
        end
        % fprintf([' --> norm(x{slab}-xk) = ' num2str(norm(x{slab}-xk))]);
        fprintf('\n');
        fprintf('        compute Ax...\n');
        if strcmp(arg.grad_type,'full')
            sAx = sAx+bss.A{slab}*x{slab};
        else
            Ab = Gblock(bss.A{slab},nblock);
            sAx = sAx+Ab{bit_reverse(iset)}*x{slab};
        end
    end
    % gather images from each node
    xcol = cell2mat(x);
    info(comm) = arg.userfun(xcol,arg.userarg{:});
    log = strcat(log,sprintf('iter %g: info = %f\\n',comm,info(comm)));
    fprintf('    ======= rmsd = %g =======\n',info(comm));
    if any(arg.isave==comm)
        fld_write([arg.path 'x_iter_' num2str(iter) '.fld' ],tom(xcol));
    end
    % figure; im('mid3',tom(xcol),[800 1200],['iter ' num2str(comm)]); cbar; pause;
    if comm==ncomm, break; end
    fprintf('    --- communication (compute residual using MPI_AllReduce) ---\n');
    if strcmp(arg.grad_type,'full')
        res = sAx-yi(:);
    else
        ia = bit_reverse(iset):nblock:Ab.odim(end);
        res = sAx-col(yi(:,:,ia));
    end
    for slab = 1:nslab
        fprintf(['    @ slab' num2str(slab) ':\n']);
        fprintf('        update zeta and g...\n');
        if strcmp(arg.grad_type,'full')
            z{slab} = bss.A{slab}'*(wi(:).*res);
        else
            Ab = Gblock(bss.A{slab},nblock);
            z{slab} = nblock*Ab{bit_reverse(iset)}'*(col(wi(:,:,ia)).*res);
        end
        g{slab} = (rho(comm)*z{slab}+g{slab})/(rho(comm)+1);
    end
    if strcmp(arg.grad_type,'partial')
        iset = next(iset);
    end
    fprintf('    --- communication (exchange end-slices with adjacent slabs) ---\n');
    for slab = 1:nslab
        fprintf(['    @ slab ' num2str(slab) ':\n']);
        if slab>1
            vt0 = vt{slab};
            ht = get_element(x{slab-1},3,reg.c_map{slab-1})-et{slab};
            ut = vt{slab};
            vtold = vt{slab};
            told = 1;
            fprintf(['        update vt... ' count2],0);
            for diter = 1:nditer
                utp = [ut; get_element(x{slab},2,reg.c_map{slab})];
                num = eta(comm)*(ut-ht)+get_element(reg.R_tc{slab}.cgrad(reg.R_tc{slab},utp),2,reg.tc_map{slab});
                den = eta(comm)+get_element(reg.R_tc{slab}.denom(reg.R_tc{slab},utp),2,reg.tc_map{slab});
                vt{slab} = ut-num./den;
                vt{slab} = max(vt{slab},voxmin);
                vt{slab} = min(vt{slab},voxmax);
                if (ut-vt{slab})'*(vt{slab}-vtold)>0
                    t = 1;
                    ut = vt{slab};
                else
                    t = (1+sqrt(1+4*told^2))/2;
                    ut = vt{slab}+(told-1)/t*(vt{slab}-vtold);
                end
                vtold = vt{slab};
                told = t;
                fprintf([back2 count2],diter);
            end
            fprintf([' --> rmsd(vt,vt0) = ' num2str(norm(vt{slab}-vt0)/sqrt(length(vt0)))]);
            fprintf('\n');
            fprintf('        update et...\n');
            et{slab} = vt{slab}-ht;
        end
        if slab<nslab
            vb0 = vb{slab};
            hb = get_element(x{slab+1},2,reg.c_map{slab+1})-eb{slab};
            ub = vb{slab};
            vbold = vb{slab};
            told = 1;
            fprintf(['        update vb... ' count2],0);
            for diter = 1:nditer
                ubp = [get_element(x{slab},3,reg.c_map{slab}); ub];
                num = eta(comm)*(ub-hb)+get_element(reg.R_cb{slab}.cgrad(reg.R_cb{slab},ubp),3,reg.cb_map{slab});
                den = eta(comm)+get_element(reg.R_cb{slab}.denom(reg.R_cb{slab},ubp),3,reg.cb_map{slab});
                vb{slab} = ub-num./den;
                vb{slab} = max(vb{slab},voxmin);
                vb{slab} = min(vb{slab},voxmax);
                if (ub-vb{slab})'*(vb{slab}-vbold)>0
                    t = 1;
                    ub = vb{slab};
                else
                    t = (1+sqrt(1+4*told^2))/2;
                    ub = vb{slab}+(told-1)/t*(vb{slab}-vbold);
                end
                vbold = vb{slab};
                told = t;
                fprintf([back2 count2],diter);
            end
            fprintf([' --> rmsd(vb,vb0) = ' num2str(norm(vb{slab}-vb0)/sqrt(length(vb0)))]);
            fprintf('\n');
            fprintf('        update eb...\n');
            eb{slab} = vb{slab}-hb;
        end
    end
end

xrec = tom(cell2mat(x));
info = [info0; info];

fid = fopen([arg.path 'recon.log'],'wt');
fprintf(fid,log);
fclose(fid);
