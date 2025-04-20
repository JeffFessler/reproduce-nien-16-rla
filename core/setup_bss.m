  function ob = setup_bss(A, nslab, varargin)
%|function ob = setup_bss(A, nslab, [options])
%|
%| setup column-wise system matrix decomposition and the corresponding
%| block separable surrogate
%|
%| system matrix = A = [A1 A2 ... Anslab]
%|
%| in
%|    A        [nd np]        Gcone object
%|    nslab                   number of slabs
%|
%| option
%|    wi       [ns nt na]     weighting sinogram (default: [] for uniform)
%|    how                     partition type (default: 'uniform')
%|
%| out
%|    ob                      bss object
%|

if nargin<2, help(mfilename), error(mfilename), end

% only xyz order is implemented
if A.zxy, fail('zxy order is not allowed'), end

% defaults
arg.wi = [];
arg.how = 'uniform';
arg = vararg_pair(arg,varargin);

% some parameters used frequently
ig0 = A.arg.ig;
cg0 = A.arg.cg;

% statistical weighting matrix
wi = arg.wi;
if isempty(wi)
    wi = ones(A.odim);
end

% construct bss object
printm 'resolve slab partition';
switch arg.how
    case 'uniform'
        nn = floor(ig0.nz/nslab);
        par = ones(1,nslab)*nn;
        ex = ig0.nz-nn*nslab;
        par(1:ex) = par(1:ex)+1;
    case 'non-uniform'
        A1 = A'*ones(A.odim,'single');%>0;
        csA1 = cumsum(squeeze(sum(sum(A1,1),2)));
        dist = abs(repmat(csA1,[1 nslab-1])-csA1(end)/nslab*repmat(1:nslab-1,[ig0.nz 1]));
        [~,idx] = min(dist,[],1);
        par = diff([0 idx ig0.nz]);
    otherwise
        fail 'bad partition how';
end
ob.iz = mat2cell(1:ig0.nz,1,par);
pi_slab = 1./par; pi_slab = pi_slab/sum(pi_slab);
% ob.gain = num2cell(1./pi_slab);
ob.A1 = A*ones(A.np,1,'single');

for slab = 1:nslab
    printm(['slab ' num2str(slab)]);
    ob.ig{slab} = image_geom('nx',ig0.nx,'fov',ig0.fov,'nz',par(slab),'dz',ig0.dz,...
        'offset_z',ig0.offset_z+(1+ig0.nz)/2-(ob.iz{slab}(1)+ob.iz{slab}(end))/2);
    %ob.ig{slab}.mask = ob.ig{slab}.circ>0;
    ob.ig{slab}.mask = ig0.mask(:,:,ob.iz{slab});
    ob.exm{slab} = @(x) x(:,:,ob.iz{slab});
    ob.tov{slab} = @(x) ob.ig{slab}.maskit(x);
    ob.tom{slab} = @(x) ob.ig{slab}.embed(x);
    ob.A{slab} = Gcone(cg0,ob.ig{slab},'type',A.type);
    % ob.dwls{slab} = ob.A{slab}'*(wi(:).*(ob.A{slab}*ones(ob.A{slab}.np,1,'single')));
end
