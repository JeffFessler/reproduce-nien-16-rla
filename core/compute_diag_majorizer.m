function [dwls,samp,cont] = compute_diag_majorizer(Ab,W,u)

nBlock = Ab.nblock;
switch length(Ab.idim)
    case 2
        extract_views = @(x,ia) x(:,ia);
    case 3
        extract_views = @(x,ia) x(:,:,ia);
    otherwise
        error 'dimension is not applicable!!!';
end

samp = zeros(size(u));
dmax = zeros(size(u));
dmin = inf(size(u));
tic;
for iblock = 1:nBlock
    ia = iblock:nBlock:W.odim(end);
    dwlsi = (Ab{iblock}'*(col(extract_views(W.arg.diag,ia)).*(Ab{iblock}*u)))./u;
    dmax = max(dmax,dwlsi);
    dmin = min(dmin,dwlsi);
    samp = samp+(dwlsi>0);
	
	fprintf('*');
	if mod(iblock,100)==0
	    fprintf('\n');
	end
end
cont = dmin./(dmax+(dmax==0));
dwls = dmax*nBlock;
tt = toc;
fprintf(' (Finish computing dwls: in %g seconds)\n',tt);

% dwls = zeros(size(u));
% tic;
% for iblock = 1:nBlock
%     ia = iblock:nBlock:Ab.odim(end);
% 	dwls = dwls+(Ab{iblock}'*(col(extract_views(W.arg.diag,ia)).*(Ab{iblock}*u)))./u;
% 	
% 	fprintf('*');
% 	if mod(iblock,100)==0
% 	    fprintf('\n');
% 	end
% end
% tt = toc;
% fprintf(' (Finish computing dwls: in %g seconds)\n',tt);
