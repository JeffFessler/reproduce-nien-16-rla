function [iVals,Ai,gi,dwls] = setup_ct_sag1(A,W,nBlock,nIter,order)

Ai = Gblock(A,nBlock,0);

lstr = length(num2str(nBlock));
count = sprintf('[%%%dd/%d]',lstr,nBlock);
back = repmat('\b',[1 1+lstr+1+lstr+1]);

dwls = 0;
gi = cell(nBlock,1);
fprintf(sprintf('Generating block projection matrices view-by-view... %s',count),0);
tic;
for iblock = 1:nBlock
    ia = iblock:nBlock:A.odim(end);
    gi{iblock} = zeros(Ai{iblock}.size(1),1,'single');
    dwls = max(dwls,nBlock*(Ai{iblock}'*(col(W.arg.diag(:,:,ia)).*(Ai{iblock}*ones(A.np,1)))));
    
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
