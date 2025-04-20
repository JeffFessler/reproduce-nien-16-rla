function x = prox_ct_with_sr1_inv_hessian(b,d,u,x0,iFixed)
% x = prox_ct_with_sr1_inv_hessian(b,d,u,x0,iFixed) solves the proximity
% operator
%    x = prox(b;d,u,x0,iFixed)
%      = argmin_z {1/2 ||z-b||_invH^2}
%          s.t.   z \in {z|[z]_i = [x0]_i if i \in iFixed; [z]_i>=0 O.W.},
% where invH = diag{d}+uu'.

if norm(u)>0
    iNneg = ~iFixed;
    r = sort((x0.*iFixed-b)./u);
    x0_minus_b = x0-b;
    v = @(r) ...
        iFixed.*...
        ((r*u-x0_minus_b).*((x0_minus_b-r*u)>=0)+...
         (r*u-x0_minus_b).*((x0_minus_b-r*u)<=0))+...
        iNneg.*...
        (r*u+b).*((b+r*u)<=0);
    f = @(r) ...
        r+u'*(v(r)./d);
    ialpha = 1; ibeta = length(r);
    falpha = f(r(ialpha)); fbeta = f(r(ibeta));
    imid = floor((ialpha+ibeta)/2);
    ii = 0;
    while ibeta-ialpha>1
        ii = ii+1;
        ff = f(r(imid));
        if ff<0
            ialpha = imid;
            falpha = ff;
        else
            ibeta = imid;
            fbeta = ff;
        end
        imid = floor((ialpha+ibeta)/2);
    end
    rr = r(ialpha)-falpha*(r(ibeta)-r(ialpha))/(fbeta-falpha);
    lambda = max((x0.*iFixed-b-rr*u)./d,0);
    mu = max((b+rr*u-x0)./d,0); mu(iNneg) = 0;
    x = max(b+(lambda-mu).*d+rr*u,0); x(iFixed) = x0(iFixed);
else
    x = max(b,0);
    x(iFixed) = x0(iFixed);
end

% r = (x0.*iFixed-b)./u;
% maxneg = -inf; fmaxneg = -inf;
% minpos = inf; fminpos = inf;
% for ii = 1:length(r)
%     v = iFixed.*...
%         ((r(ii)*u-(x0-b)).*((x0-b-r(ii)*u)>=0)+...
%          (r(ii)*u-(x0-b)).*((x0-b-r(ii)*u)<=0))+...
%         iNneg.*...
%         (r(ii)*u+b).*((b+r(ii)*u)<=0);
%     f = r(ii)+u'*(v./d);
%     if f>=0 && r(ii)<minpos && f<=fminpos
%         minpos = r(ii);
%         fminpos = f;
%     elseif f<=0 && r(ii)>maxneg && f>=fmaxneg
%         maxneg = r(ii);
%         fmaxneg = f;
%     end
% end
% rr = maxneg-fmaxneg*(minpos-maxneg)/(fminpos-fmaxneg);
% lambda = max((x0.*iFixed-b-rr*u)./d,0);
% mu = max((b+rr*u-x0)./d,0); mu(iNneg) = 0;
% x = max(b+(lambda-mu).*d+rr*u,0); x(iFixed) = x0(iFixed);