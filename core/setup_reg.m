  function ob = setup_reg(R, iz)
%|function ob = setup_reg(R, iz)
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
%| top = 2, center = 1, bottom = 3

if nargin<2, help(mfilename), error(mfilename), end

% only xyz order is implemented
if R.offsets_is_zxy, fail('zxy order is not allowed'), end

% some parameters used frequently
kappa = R.cdp_arg{1};
mask = R.mask;
beta = R.cdp_arg{3};
pot_arg = {R.cdp_arg{4},R.cdp_arg{5}(1),R.cdp_arg{5}(2:end)};
dist_pow = R.distance_power;
nthread = R.nthread;
nslab = length(iz);

for slab = 1:nslab
    printm(['slab ' num2str(slab)]);
    izp = iz{slab};
    pad_top = slab>1;
    pad_bottom = slab<nslab;
    if pad_top, izp = [iz{slab-1}(end) izp]; end
    if pad_bottom, izp = [izp iz{slab+1}(1)]; end

    ob.c_map{slab} = single(mask(:,:,iz{slab}));
    ob.c_map{slab}(:,:,1) = ob.c_map{slab}(:,:,1)*2;
    ob.c_map{slab}(:,:,end) = ob.c_map{slab}(:,:,end)*3;
    
    ob.R_tcb{slab} = Reg1(kappa(:,:,izp),'offsets','3d:26','beta',beta,...
        'type_penal','mex','nthread',jf('ncore'),'distance_power',dist_pow,...
        'mask',mask(:,:,izp),'pot_arg',pot_arg);
    ob.tcb_map{slab} = single(mask(:,:,izp));
    if pad_top, ob.tcb_map{slab}(:,:,1) = ob.tcb_map{slab}(:,:,1)*2; end
    if pad_bottom, ob.tcb_map{slab}(:,:,end) = ob.tcb_map{slab}(:,:,end)*3; end

    if pad_top
        ob.R_tc{slab} = Reg1(kappa(:,:,izp(1:2)),'offsets','3d:26','beta',beta,...
            'type_penal','mex','nthread',jf('ncore'),'distance_power',dist_pow,...
            'mask',mask(:,:,izp(1:2)),'pot_arg',pot_arg);
        ob.tc_map{slab} = single(mask(:,:,izp(1:2)));
        ob.tc_map{slab}(:,:,1) = ob.tc_map{slab}(:,:,1)*2;
    else
        ob.R_tc{slab} = [];
        ob.tc_map{slab} = [];
    end

    if pad_bottom
        ob.R_cb{slab} = Reg1(kappa(:,:,izp(end-1:end)),'offsets','3d:26','beta',beta,...
            'type_penal','mex','nthread',jf('ncore'),'distance_power',dist_pow,...
            'mask',mask(:,:,izp(end-1:end)),'pot_arg',pot_arg);
        ob.cb_map{slab} = single(mask(:,:,izp(end-1:end)));
        ob.cb_map{slab}(:,:,end) = ob.cb_map{slab}(:,:,end)*3;
    else
        ob.R_cb{slab} = [];
        ob.cb_map{slab} = [];
    end
end
