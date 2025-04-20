  function ob = setup_reg0(R, iz)
%|function ob = setup_reg0(R, iz)
%|
%| setup (overlapped) slab-partitioned regularizers
%|
%| in
%|    R                       penalty object (see Reg1.m)
%|    iz                      the z-index of each slab
%|
%| out
%|    ob                      reg object
%|

if nargin<2, help(mfilename), error(mfilename), end

% only xyz order is implemented
if R.offsets_is_zxy, fail('zxy order is not allowed'), end

% some parameters used frequently
kappa = R.cdp_arg{1};
mask = R.mask;
beta = R.cdp_arg{3};
pot_arg = {R.cdp_arg{4},R.cdp_arg{5}(1),R.cdp_arg{5}(2:end)};
dist_pow = R.distance_power;
nx = R.dim(1);
ny = R.dim(2);
nz = R.dim(3);
nslab = length(iz);

for slab = 1:nslab
    printm(['slab ' num2str(slab)]);
    izp = iz{slab};
    pad_top = slab>1;%iz{slab}(1)>1;
    pad_bottom = slab<nslab;%iz{slab}(end)<nz;
    if pad_top, izp = [iz{slab-1}(end) izp]; end
    if pad_bottom, izp = [izp iz{slab+1}(1)]; end
    
    pad_top0 = 1; pad_bottom0 = 1; % always pad in tc and cb for using 3d:26
    ob.R_tcb{slab} = Reg1(kappa(:,:,izp),'offsets','3d:26','beta',beta,...
        'type_penal','mex','nthread',jf('ncore'),'distance_power',dist_pow,...
        'mask',mask(:,:,izp),'pot_arg',pot_arg);
    ob.R_tc{slab} = Reg1(kappa(:,:,izp(1:1+pad_top0)),'offsets','3d:26','beta',beta,...
        'type_penal','mex','nthread',jf('ncore'),'distance_power',dist_pow,...
        'mask',mask(:,:,izp(1:1+pad_top0)),'pot_arg',pot_arg);
    ob.R_cb{slab} = Reg1(kappa(:,:,izp(end-pad_bottom0:end)),'offsets','3d:26','beta',beta,...
        'type_penal','mex','nthread',jf('ncore'),'distance_power',dist_pow,...
        'mask',mask(:,:,izp(end-pad_bottom0:end)),'pot_arg',pot_arg);
    % operate on slab
    ob.tov{slab} = @(x) x(mask(:,:,iz{slab}));
    ob.tom{slab} = @(x) embed(x,mask(:,:,iz{slab}));
    ob.ttov{slab} = @(x) x(mask(:,:,iz{slab}(1)));
    ob.ttom{slab} = @(x) embed(x,mask(:,:,iz{slab}(1)));
    ob.btov{slab} = @(x) x(mask(:,:,iz{slab}(end)));
    ob.btom{slab} = @(x) embed(x,mask(:,:,iz{slab}(end)));
    ob.mtop{slab} = @(x) x(:,:,1);
    ob.mbottom{slab} = @(x) x(:,:,end);
    ob.vtop{slab} = @(x) ob.ttov{slab}(ob.mtop{slab}(ob.tom{slab}(x)));
    ob.vbottom{slab} = @(x) ob.btov{slab}(ob.mbottom{slab}(ob.tom{slab}(x)));
    ob.msett{slab} = @(x,t) cat(3,t,x(:,:,2:end));
    ob.vsett{slab} = @(x,t) ob.tov{slab}(ob.msett{slab}(ob.tom{slab}(x),ob.ttom{slab}(t)));
    ob.msetb{slab} = @(x,b) cat(3,x(:,:,1:end-1),b);
    ob.vsetb{slab} = @(x,b) ob.tov{slab}(ob.msetb{slab}(ob.tom{slab}(x),ob.btom{slab}(b)));
    % operate on padded slab
    % - convert to vector
    ob.tov_tcb{slab} = @(x) x(ob.R_tcb{slab}.mask);
    ob.tov_tc{slab} = @(x) x(ob.R_tc{slab}.mask);
    ob.tov_cb{slab} = @(x) x(ob.R.cb{slab}.mask);
    % - convert to matrix
    ob.tom_tcb{slab} = @(x) embed(x,ob.R_tcb{slab}.mask);
    ob.tom_tc{slab} = @(x) embed(x,ob.R_tc{slab}.mask);
    ob.tom_cb{slab} = @(x) embed(x,ob.R_cb{slab}.mask);
    % - convert top pad
    if pad_top 
        %ob.ttov_tcb{slab} = @(x) x(ob.R_tcb{slab}.mask(:,:,1));
        ob.ttov_tc{slab} = @(x) x(ob.R_tc{slab}.mask(:,:,1));
        %ob.ttom_tcb{slab} = @(x) embed(x,ob.R_tcb{slab}.mask(:,:,1));
        ob.ttom_tc{slab} = @(x) embed(x,ob.R_tc{slab}.mask(:,:,1));
    else
        %ob.ttov_tcb{slab} = @(x) fail('invalid operation');
        ob.ttov_tc{slab} = @(x) fail('invalid operation');
        %ob.ttom_tcb{slab} = @(x) fail('invalid operation');
        ob.ttom_tc{slab} = @(x) fail('invalid operation');
    end
    % - convert bottom pad
    if pad_bottom
        %ob.btov_tcb{slab} = @(x) x(ob.R_tcb{slab}.mask(:,:,end));
        ob.btov_cb{slab} = @(x) x(ob.R_tc{slab}.mask(:,:,end));
        %ob.btom_tcb{slab} = @(x) embed(x,ob.R_tcb{slab}.mask(:,:,end));
        ob.btom_cb{slab} = @(x) embed(x,ob.R_tc{slab}.mask(:,:,end));
    else
        %ob.btov_tcb{slab} = @(x) fail('invalid operation');
        ob.btov_cb{slab} = @(x) fail('invalid operation');
        %ob.btom_tcb{slab} = @(x) fail('invalid operation');
        ob.btom_cb{slab} = @(x) fail('invalid operation');
    end
    % - extract portion
    if pad_top
        %ob.mtop_tcb{slab} = @(x) x(:,:,1);
        ob.mtop_tc{slab} = @(x) x(:,:,1);
        %ob.vtop_tcb{slab} = @(x) ob.ttov_tcb{slab}(ob.mtop_tcb{slab}(ob.tom_tcb{slab}(x)));
        ob.vtop_tc{slab} = @(x) ob.ttov_tc{slab}(ob.mtop_tc{slab}(ob.tom_tc{slab}(x)));
    else
        %ob.mtop_tcb{slab} = @(x) fail('invalid operation');
        ob.mtop_tc{slab} = @(x) fail('invalid operation');
        %ob.vtop_tcb{slab} = @(x) fail('invalid operation');
        ob.vtop_tc{slab} = @(x) fail('invalid operation');
    end
    if pad_bottom
        %ob.mbottom_tcb{slab} = @(x) x(:,:,end);
        ob.mbottom_cb{slab} = @(x) x(:,:,end);
        %ob.vbottom_tcb{slab} = @(x) ob.btov_tcb{slab}(ob.mbottom_tcb{slab}(ob.tom_tcb{slab}(x)));
        ob.vbottom_tc{slab} = @(x) ob.btov_tc{slab}(ob.mbottom_tc{slab}(ob.tom_tc{slab}(x)));
    else
        %ob.mbottom_tcb{slab} = @(x) fail('invalid operation');
        ob.mbottom_cb{slab} = @(x) fail('invalid operation');
        %ob.vbottom_tcb{slab} = @(x) fail('invalid operation');
        ob.vbottom_cb{slab} = @(x) fail('invalid operation');
    end
    ob.mcenter_tcb{slab} = @(x) x(:,:,1+pad_top:end-pad_bottom);
    %ob.mcenter_tc{slab} = @(x) x(:,:,1+pad_top:end);
    %ob.mcenter_cb{slab} = @(x) x(:,:,1:end-pad_bottom);
    ob.vcenter_tcb{slab} = @(x) ob.tov{slab}(ob.mcenter_tcb{slab}(ob.tom_tcb{slab}(x)));
end
