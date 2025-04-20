function [forwi,backi,iVals,gi,denom] = setup_ct_sag(A,W,nBlock,nIter,order)

Ai = Gblock(A,nBlock,0);

switch length(A.idim)
    case 2
        extract_views = @(x,ia) x(:,ia);
    case 3
        extract_views = @(x,ia) x(:,:,ia);
    otherwise
        error 'dimension is not applicable!!!';
end

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

denom = 0;
Wi = cell(nBlock,1); forwi = cell(nBlock,1); backi = cell(nBlock,1); gi = cell(nBlock,1);
fprintf(sprintf('Generating block projection matrices view-by-view... %s',count),0);
tic;
for iblock = 1:nBlock
    ia = iblock:nBlock:A.odim(end);
    if strcmp(W.caller,'Gdiag')
        Wi{iblock} = Gdiag(extract_views(W.arg.diag,ia));
    elseif strcmp(W.caller,'Gweight')
        Wi{iblock} = Gweight(extract_views(W.arg.diag,ia),extract_views(W.arg.Kc,ia),W.arg.rho);
    else
        error 'unknown W.caller!!!';
    end
	forwi{iblock} = @(x,y) nBlock*Wi{iblock}*(Ai{iblock}*x-col(extract_views(y,ia)));
	backi{iblock} = @(y) Ai{iblock}'*y;
    gi{iblock} = zeros(Ai{iblock}.size(1),1,'single');
    denom = max(denom,nBlock*(Ai{iblock}'*(Wi{iblock}*(Ai{iblock}*ones(A.np,1)))));
    
    fprintf([back count],iblock);
end
tt = toc;
fprintf(' (in %g seconds)\n',tt);

switch order
    case 'rand'
        iVals = randi(nBlock,1,nIter*nBlock);
    case 'bit-rev'
        iVals = repmat(subset_start(nBlock)',1,nIter);
    case 'seq'
        iVals = repmat((1:nBlock),1,nIter);
    otherwise
        iVals = randi(nBlock,1,nIter*nBlock);
end
