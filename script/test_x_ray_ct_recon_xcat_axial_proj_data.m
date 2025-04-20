% test_x_ray_ct_recon_xcat_axial_proj_data.m
close all; clear all; clc;

%% setup geometry, image, and sinogram
printm 'setup image and ct geometry...';
f.in = './in/';
f.out = './out/';
ig_hi = image_geom('nx',1024,'fov',500,'nz',154,'dz',0.625);
ig_hi.mask = ig_hi.circ>0;
cg = ct_geom('ge2');
% figure; cg.plot3(ig_hi);

printm 'generate noiseless projection data...';
proj_type = 'sf2';
A_hi = Gcone(cg,ig_hi,'type',proj_type);
file = '/n/ir62/z/hungnien/xcat_high_res/xtrue_hi.fld';
x_hi = fld_read(file);
% figure; im('mid3',x_hi,[800 1200]); cbar;
ytrue = A_hi*x_hi;
% figure; im(ytrue(:,floor(end/2),:)); cbar;

printm 'generate noisy projection data...';
b0 = 1e5;
HU2unit = 0.04/2000;
rng(0);
bi = poisson(b0*exp(-ytrue*HU2unit),0,'factor',0.4);    % convert to 1/mm units
if any(bi(:) == 0)
    warn('%d of %d values are 0 in sinogram!',sum(bi(:)==0),length(bi(:)));
end
y = log(b0./max(bi,1))/HU2unit;
fld_write([f.in 'yi-tsa.fld'],y);
% figure; im(y(:,floor(end/2),:)); cbar;

%% setup target geometry and fbp
printm 'setup target image geometry...';
ig = image_geom('nx',512,'fov',500,'nz',90,'dz',0.625);
ig.mask = ig.circ>0;
A = Gcone(cg,ig,'type',proj_type);
% figure; cg.plot3(ig);
printm 'generate initial fbp image...';
% xini = feldkamp(cg,ig,y,'w1cyl',1);
% xini = feldkamp(cg,ig,y,'window','hanning,0.8','w1cyl',1);
xini = feldkamp(cg,ig,y,'window','hanning,0.8','w1cyl',1,'extrapolate_t',round(1.3*cg.nt/2));
[xx,yy] = ndgrid(-255.5:1:255.5,-255.5:1:255.5);
roi2 = (xx.^2/230^2+(yy-10).^2/170^2)<1;
clear xx yy;
npad = 13;
start_slice = npad+1; end_slice = ig.nz-npad;
fld_write([f.in 'xini.fld'],xini);
% figure; im('mid3',xini,[800 1200]); cbar;

%% setup statistical weighting and kappa
printm 'setup weighting matrix...';
w = exp(-y*HU2unit);
fld_write([f.in 'wi-tsa.fld'],w);
% figure; im(w(:,floor(end/2),:)); cbar;
printm 'compute voxel-dependent kappa...';
kappa = sqrt(div0(A'*w,A'*ones(size(w))));
fld_write([f.in 'kappa.fld'],kappa);
% figure; im('mid3',kappa); cbar;

%% save projector parameters
printm 'save projector parameters...';
save([f.in 'proj_param.mat'],'ig','cg','proj_type','start_slice','end_slice','roi2');
