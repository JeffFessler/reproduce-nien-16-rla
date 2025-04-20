% test_x_ray_ct_recon_xcat_axial_generate_xref.m
if 0
    test_x_ray_ct_recon_xcat_axial_proj_data
    f.in = './in/';
    f.out = './out/';
else
    close all; clear all; clc;
    f.in = './in/';
    f.out = './out/';
    load([f.in 'proj_param.mat']);
    A = Gcone(cg,ig,'type',proj_type);
    y = fld_read([f.in 'yi-tsa.fld']);
    w = fld_read([f.in 'wi-tsa.fld']);
    kappa = fld_read([f.in 'kappa.fld']);
    xini = fld_read([f.in 'xini.fld']);
end
figure; im('mid3',xini(:,:,start_slice:end_slice),[800 1200]); cbar; axx = gca;

%% setup edge-preserving regularizer
printm 'setup edge-preserving regularizer...';
% delta = 1e1; pot = {'lange3',delta}; gain = 2^10;
delta = 1e1; pot = {'lange3',delta}; gain = 2^9;	% under-regularized
% delta = 1e0; pot = {'lange3',delta}; gain = 2^13;
% delta = 1e0; pot = {'lange3',delta}; gain = 2^12;	% under-regularized
b1 = 1/ig.dx^2; b2 = 1/(ig.dx^2+ig.dy^2);
b3 = 1/ig.dz^2; b4 = 1/(ig.dx^2+ig.dz^2);
b5 = 1/(ig.dx^2+ig.dy^2+ig.dz^2);
beta = gain*[b1 b1 b2 b2 b5 b4 b5 b4 b3 b4 b5 b4 b5];
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
save([f.in 'reg_param_delta_' num2str(delta) '_under.mat'],'pot','beta');

%% setup useful anonymous functions
printm 'setup useful anonymous functions...';
matx2vecx = @(matx) (matx(R.mask(:)));
vecx2matx = @(vecx) (embed(vecx,R.mask));
maty2vecy = @(maty) (maty(:));
vecy2maty = @(vecy) (reshape(vecy,size(y)));

%% generate reference reconstruction
nIter = 10;
nBlock = 12;
iROI = roi2(:,:,ones(ig.nz,1)); iROI(:,:,1:start_slice-1) = 0; iROI(:,:,end_slice+1:end) = 0;
iSave = [];
sDir = '';
xos = ct_os_nes05_mc(y,A,Gdiag(w),R,xini,nIter,nBlock,xini,iROI,iSave,sDir);
% figure; im('mid3',xos(:,:,start_slice:end_slice),[800 1200]); cbar; axx = [axx gca];
% linkaxes(axx);
nIter = 2000;
nBlock = 1;
iSave = 0:200:nIter;
sDir = '';
xref = ct_os_nes83_mc(y,A,Gdiag(w),R,xos,nIter,nBlock,xos,iROI,iSave,sDir);
fld_write([f.out 'xref_delta_' num2str(delta) '_under.fld'],xref);
figure; im('mid3',xref(:,:,start_slice:end_slice),[800 1200]); cbar; axx = [axx gca];
linkaxes(axx);

exit;

