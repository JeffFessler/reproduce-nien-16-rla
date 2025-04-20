% function denom = compute_reg_sqs(R,x,u)
function denom = compute_reg_sqs(R,w,x,u)

% xmat = embed(x,R.mask);
% umat = embed(u,R.mask);
% 
% denom = 0;
% for mm = 1:numel(R.C1s)
% 	Cm = R.C1s{mm};
% 	d = Cm*xmat;
% 	Cm = abs(Cm);
% 	ck = Cm*umat; % |C|*u
% 	pot = R.pot{mm};
% 	wt = pot.wpot(d); % potential function Huber curvatures
% 	wt = wt.*reshape(R.wt.col(mm),size(wt));
% 	tmp = Cm'*(wt.*ck);
% 	denom = denom+tmp;
% end
% 
% denom = denom(R.mask(:))./u;


C = R.C1;
d = C*x;
absC = abs(C);
ck = absC*u;
pot = R.pot{1};
wt = pot.wpot(d);
% wt = wt.*compute_reg_wt(R);
wt = wt.*w;
denom = (absC'*(wt.*ck))./u;
