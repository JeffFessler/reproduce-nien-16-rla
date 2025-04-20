% test_lasso_primal_dual_gap.m
close all; clear all; clc;
warning off;
% rng(0);

%% solve sparse linear regression using cvx
printm 'solve sparse linear regression using cvx';
% m = 50; n = 200; s = 10;
m = 100; n = 400; s = 20;
% m = 500; n = 2000; s = 100;
A = randn(m,n); dA = max(eig(A'*A));
xtrue = randn(n,1).*(randperm(n)<=s)';
sigma = sqrt(0.1);
y =A*xtrue+randn(m,1)*sigma;
lambda = 1;
cvx_precision('best');
cvx_begin
    variables uh(m) xh(n);
    dual variable zh;
    minimize((y-uh)'*(y-uh)/2+lambda*norm(xh,1));
    subject to
        zh : uh==A*xh;
cvx_end

R = separableReg1d(ones(size(xh)),lambda,{'l1'});
printm 'refine solution';
[xh,uh,zh] = rlalm_lasso(xh,uh,zh,A,y,R,10000,dA,1.8,0.1,xh,uh,zh);

% x0 = zeros(n,1);
% x0 = randn(n,1);
x0 = A\y;
u0 = A*x0;
z0 = y-u0;

r_list = [0.5 0.1];
a_list = [1 1.999];

niter = 1e4;

%% solve sparse linear regression using lalm
printm 'solve sparse linear regression using lalm';
xmark1 = 10.^(0:log10(niter));
xmark2 = 0:(niter/5):niter;
id = 1; ip = 1;

for rho = r_list
    for alpha = a_list
        [~,~,~,gap_itr,gap_erg,const] = rlalm_lasso(x0,u0,z0,A,y,R,niter,dA,alpha,rho,xh,uh,zh);

        if alpha==1.999
            str = ['\alpha \approx 2, \rho = ' num2str(rho)];
        else
            str = ['\alpha = ' num2str(alpha) ', \rho = ' num2str(rho)];
        end

        gap1{id} = [const./(0:niter+eps)'];
        lgd1{id} = ['Bound (' str ')'];
        lst1{id} = ':';
        lcl1{id} = get_line_color(ip);
        mst1{id} = 'none';
        gap2{id} = gap_erg;
        lgd2{id} = ['E-gap (' str ')'];%['Ergodic (' str ')'];
        lst2{id} = '-';
        lcl2{id} = get_line_color(ip);
        mst2{id} = 'none';
        id = id+1;

        gap1{id} = gap_erg;
        lgd1{id} = ['E-gap (' str ')'];%['Ergodic (' str ')'];
        lst1{id} = '-';
        lcl1{id} = get_line_color(ip);
        mst1{id} = 'none';
        gap2{id} = gap_itr;
        lgd2{id} = ['NE-gap (' str ')'];%['Non-ergodic (' str ')'];
        lst2{id} = '--';
        lcl2{id} = get_line_color(ip);
        mst2{id} = 'none';
        id = id+1;

        ip = ip+1;
    end
end

%% plot curves
printm 'plot curves';
show_curve(0:niter,gap1,lgd1,'line_style',lst1,'marker_type',mst1,'line_color',lcl1,'scale','loglog','font_size',11);
xlabel('Number of iterations'); ylabel('Primal-dual gap');
set(gca,'XTick',xmark1,'XLim',[0 niter]);
epswrite('lasso_gap_bound.eps');

show_curve(0:niter,gap2,lgd2,'line_style',lst2,'marker_type',mst2,'line_color',lcl2,'scale','semilogy','font_size',11);
xlabel('Number of iterations'); ylabel('Primal-dual gap');
set(gca,'XTick',xmark2,'XLim',[0 niter],'YLim',[1e-15 1e3],'YTick',10.^(-15:3:3));
epswrite('lasso_gap_nonergodic.eps');
return

%% solve sparse linear regression using lalm
printm 'solve sparse linear regression using lalm';
niter = 10000; diter = niter/5;
clear gap lgd lst lcl mst;
id = 1; ip = 1;

for rho = r_list
    for alpha = a_list
        [~,~,~,gap_itr,gap_erg] = rlalm_lasso(x0,u0,z0,A,y,R,niter,dA,alpha,rho,xh,uh,zh);

        if alpha==1.999
            str = ['\alpha \approx 2, \rho = ' num2str(rho)];
        else
            str = ['\alpha = ' num2str(alpha) ', \rho = ' num2str(rho)];
        end

        gap{id} = gap_erg;
        lgd{id} = ['E-gap (' str ')'];
        lst{id} = '-';
        lcl{id} = get_line_color(ip);
        mst{id} = 'none';
        id = id+1;
        
        gap{id} = gap_itr;
        lgd{id} = ['NE-gap (' str ')'];
        lst{id} = '--';
        lcl{id} = get_line_color(ip);
        mst{id} = 'none';%get_marker(ip);
        id = id+1;

        ip = ip+1;
    end
end

%% plot curves
printm 'plot curves';
% show_curve(0:niter,gap,lgd,'xmark',0:diter:niter,'line_style',lst,'marker_type',mst,'line_color',lcl,'scale','semilogy','font_size',11,'location','northeast');
show_curve(0:niter,gap,lgd,'line_style',lst,'marker_type',mst,'line_color',lcl,'scale','semilogy','font_size',11,'location','northeast');
xlabel('Number of iterations'); ylabel('Primal-dual gap');
set(gca,'XTick',0:diter:niter,'XLim',[0 niter]);
epswrite('lasso_gap_nonergodic.eps');
