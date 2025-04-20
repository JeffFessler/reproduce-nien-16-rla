function x = prox_ct_box(y,d,u,xFBP,iFBP)

r = sort((xFBP(u~=0).*iFBP(u~=0)-y(u~=0))./u(u~=0));
g = @(r) r+u'*(iFBP.*(r*u-xFBP+y)./d+(1-iFBP).*min(r*u+y,0)./d);

ia = 1; fa = g(r(ia));  % fa <= 0
ib = length(r); fb = g(r(ib));  % fb >= 0

if fa>0 || fb<0
    % error('something is wrong here [ra = %g, fa = %g, rb = %g, fb = %g]?!',r(ia),fa,r(ib),fb);
    error('something is wrong here [fa = %g, fb = %g]?!',fa,fb);
end

while ib-ia>1
    im = floor((ia+ib)/2); fm = g(r(im));
    if fa*fm<0
        ib = im; fb = fm;
    else
        ia = im; fa = fm;
    end
end

r0 = r(ia)-fa*(r(ib)-r(ia))/(fb-fa);
x = y+d.*(max((xFBP.*iFBP-y-r0*u)./d,0)-max((y+r0*u-xFBP)./d,0).*iFBP)+r0*u;