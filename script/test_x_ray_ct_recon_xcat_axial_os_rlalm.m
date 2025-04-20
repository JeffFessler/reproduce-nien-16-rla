% test_x_ray_ct_recon_xcat_axial_os_rlalm.m
close all; clear all; clc;

%% setup projector and load recon. data
printm 'setup projector and load recon. data...';
f.in = './in/';
f.out = './out/';
load([f.in 'proj_param.mat']);
A = Gcone(cg,ig,'type',proj_type);
y = fld_read([f.in 'yi-tsa.fld']);
w = fld_read([f.in 'wi-tsa.fld']);
kappa = fld_read([f.in 'kappa.fld']);
xini = fld_read([f.in 'xini.fld']);
figure; im('mid3',xini(:,:,start_slice:end_slice),[800 1200]); cbar;

%% setup edge-preserving regularizer and load reference recon.
printm 'setup edge-preserving regularizer and load reference recon...';
delta = 1e1;
reg_force = '';
% reg_force = '_under';
load([f.in 'reg_param_delta_' num2str(delta) reg_force '.mat']);
R = Reg1(...
    kappa,...
    'offsets','3d:26',...
    'beta',beta,...
    'type_penal','mex',...
    'nthread',jf('ncore'),...
    'distance_power',0,...
    'mask',ig.mask,...
    'pot_arg',pot...
    );
xref = fld_read([f.out 'xref_delta_' num2str(delta) reg_force '.fld']);

%% setup recon. parameters
printm 'setup recon. parameters...';
niter = 20;
diter = 2;
nblock = 12;
Ab = Gblock(A,nblock);
isave = [];
path = './';
clim = [800 1200];
idx1 = 1:512; idx2 = 1+100:512-100; idx3 = start_slice:end_slice;
np = sum(ig.mask(:));
% roi = ig.mask;
roi = roi2(:,:,ones(ig.nz,1));
roi(:,:,1:start_slice-1) = 0; roi(:,:,end_slice+1:end) = 0; roi = roi(ig.mask);
fx = @(x) norm((x-xref(ig.mask)).*roi)/sqrt(sum(roi)); % RMSD
r1 = '@(k) pi/k*sqrt(1-(pi/(2*k))^2)*(k>1)+(k==1)';
r2 = '@(k) pi/(2*k)*sqrt(1-(pi/(2*(2*k)))^2)*(k>1)+(k==1)';
t0 = '@(k) max(4/(k+3),0.1)';
t1 = '@(k) 1/k';
rho0 = 1;
mu0 = 10;
tau = 2;
beta = 1.5;
thd = 0;
% dwls = 'max';
dwls = A'*(w(:).*(A*ones(np,1)));
r0 = 0.01*median(dwls);

id = 1; axx = [];

%% image recon: os-fgm
if 1
    printm 'image recon: os-fgm';
    [xrec,info] = ct_os_fgm(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'momentum','std','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-FGM2';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-nes88
if 0
    printm 'image recon: os-nes88';
    [xrec,info] = ct_os_nes88(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'average','no','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-Nes88';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: aos-nes88
if 0
    printm 'image recon: aos-nes88';
    [xrec,info] = ct_os_nes88(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'average','yes','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'aOS-Nes88';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-lalm
if 1
    printm 'image recon: os-lalm';
    [xrec,info] = ct_os_rlalm(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',r1,'alpha',1,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-LALM';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-lalm-aps
if 0
    printm 'image recon: os-lalm-aps';
    [xrec,info] = ct_os_rlalm_aps(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho0',rho0,'mu0',mu0,'tau',tau,'beta',beta,'thd',thd,'alpha',1,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-LALM (with APS)';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-ogm
if 1
    printm 'image recon: os-ogm';
    [xrec,info] = ct_os_fgm(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'momentum','opt','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-OGM2';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-rcp
if 0
    printm 'image recon: os-rcp';
    [xrec,info] = ct_os_rcp(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',r2,'alpha',1.999,'corr',0,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'Relaxed OS-CP';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-rcp (with correction)
if 0
    printm 'image recon: os-rcp (with correction)';
    [xrec,info] = ct_os_rcp(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',r2,'alpha',1.999,'corr',1,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'Relaxed OS-CP (corrected)';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-rdcp
if 0
    printm 'image recon: os-rdcp';
    [xrec,info] = ct_os_rdcp(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',r2,'alpha',1.999,'corr',0,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'Relaxed OS-DCP';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-rlalm
if 1
    printm 'image recon: os-rlalm';
    [xrec,info] = ct_os_rlalm(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',r2,'alpha',1.999,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'Relaxed OS-LALM';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-rlalm1
if 0
    printm 'image recon: os-rlalm1';
    [xrec,info] = ct_os_rlalm1(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',r2,'alpha',1.999,'theta',t0,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'Relaxed OS-LALM (with interp)';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-rlalm-aps
if 0
    printm 'image recon: os-rlalm-aps';
    [xrec,info] = ct_os_rlalm_aps(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho0',rho0,'mu0',mu0,'tau',tau,'beta',beta,'thd',thd,'alpha',1.999,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'Relaxed OS-LALM (with APS)';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: spdc
if 0
    printm 'image recon: spdc';
    [xrec,info] = ct_spdc(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'order','fft','rho',0.05,'theta',1,'rate',1,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'SPDC';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-gsm
if 0
    printm 'image recon: os-gsm';
    [xrec,info] = ct_os_gsm(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-GSM';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-fgm1
if 0
    printm 'image recon: os-fgm1';
    [xrec,info] = ct_os_fgm1(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'momentum','std','average','no','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-FGM1';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: aos-fgm1
if 0
    printm 'image recon: aos-fgm1';
    [xrec,info] = ct_os_fgm1(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'momentum','std','average','yes','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'aOS-FGM1';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-ogm1
if 0
    printm 'image recon: os-ogm1';
    [xrec,info] = ct_os_fgm1(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'momentum','opt','average','no','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'OS-OGM1';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: aos-ogm1
if 0
    printm 'image recon: aos-ogm1';
    [xrec,info] = ct_os_fgm1(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'momentum','opt','average','yes','userfun',fx);
    x{id} = xrec;
    rmsd{id} = info;
    legd{id} = 'aOS-OGM1';
    msty{id} = get_marker(id);
    lsty{id} = '-';
    lcol{id} = get_line_color(id);
    xt{id} = xrec(idx1,idx2,end/2);
    dt{id} = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x{id},clim,legd{id}); cbar; axx = [axx gca];
    id = id+1;
end

%% image recon: os-sqs
if 1
    printm 'image recon: os-sqs';
    [xrec,info] = ct_os_rlalm(xini,Ab,y,R,'wi',w,'niter',niter,'isave',isave,'path',path,'denom',dwls,'rho',1,'alpha',1,'userfun',fx);
    x_sqs = xrec;
    rmsd_sqs = info;
    legd_sqs = 'OS-SQS';
    msty_sqs = 'none';
    lsty_sqs = ':';
    lcol_sqs = [0 0 0];
    xt_sqs = xrec(idx1,idx2,end/2);
    dt_sqs = xrec(idx1,idx2,end/2)-xref(idx1,idx2,end/2);
    % figure; im('mid3',x_sqs,clim,legd_sqs); cbar; axx = [axx gca];
end
% save('xcat_proposed_12_blocks_20_iters.mat','x','legd','xini','xref','-v7.3');
% save('xcat_cmp_12_blocks_10_iters.mat','x','legd','x_sqs','legd_sqs','-v7.3');
save('xcat_cmp_12_blocks_20_iters.mat','x','legd','x_sqs','legd_sqs','-v7.3');
return

%% plot figures
line_width = 2;
font_size = 11;

if ~isempty(axx)
    linkaxes(axx);
end

rr = rmsd; rr{end+1} = rmsd_sqs;
ll = legd; ll{end+1} = legd_sqs;
ss = lsty; ss{end+1} = lsty_sqs;
cc = lcol; cc{end+1} = lcol_sqs;
mm = msty; mm{end+1} = msty_sqs;
h = show_curve(0:niter,rr,ll,'xmark',0:diter:niter,'line_style',ss,'marker_type',mm,'line_color',cc);
set(gca,'XTick',0:diter:niter);
hold on; line([0 niter],[1 1],'LineStyle','--','Color','k'); hold off;
text(niter/100,1,'1 HU','HorizontalAlignment','left','VerticalAlignment','bottom','FontSize',8);
hold on; line([0 niter],[5 5],'LineStyle','--','Color','k'); hold off;
text(niter/100,5,'5 HU','HorizontalAlignment','left','VerticalAlignment','bottom','FontSize',8);
xlabel('Number of iterations'); ylabel('RMS difference [HU]');
epswrite('test_epswrite.eps');

return
plot_image(xt,legd,2,2,800:100:1200,'w',18);
plot_image(dt,legd,2,2,-50:25:50,'w',18);

if isvar('x')
    % plot rmsd curves
    h_fig = plot_curve1(0:niter,rmsd,0:diter:niter,legd,msty,lsty,lcol,line_width,font_size);
    set(gca,'XTick',0:diter:niter);
    hold on; line([0 niter],[1 1],'LineStyle','--','Color','k'); hold off;
    text(niter/100,1,'1 HU','HorizontalAlignment','left','VerticalAlignment','bottom','FontSize',8);
    hold on; line([0 niter],[5 5],'LineStyle','--','Color','k'); hold off;
    text(niter/100,5,'5 HU','HorizontalAlignment','left','VerticalAlignment','bottom','FontSize',8);
    xlabel('Number of iterations'); ylabel('RMS difference [HU]');
    % save2eps(['rmsd_' num2str(nblock) '_blocks.eps']);
    % save(['recon_' num2str(nblock) '_blocks.mat'],'x','rmsd','legd','-v7.3');

    %xc{1} = xini(idx1,idx2,floor(end/2)); tc{1} = 'FBP';
    %xc{2} = xref(idx1,idx2,floor(end/2)); tc{2} = 'Converged';
    %xc{3} = x{1}(idx1,idx2,floor(end/2)); tc{3} = 'Proposed';
    %plot_image(xc,tc,1,3,800:100:1200,'w',11);
    %save2eps('xcat_fbp_conv_prop.eps');
    %save('xcat_recon.mat','x','xc','tc','rmsd','legd','-v7.3');
end
if isvar('x_sqs')
    if isvar('h_fig')
        hold on; plot(0:niter,rmsd_sqs,'LineStyle','-.','Color','k','LineWidth',line_width); hold off;
    end
    % save(['recon_sqs_' num2str(nblock) '_blocks.mat'],'x_sqs','rmsd_sqs','legd_sqs','-v7.3');
end
