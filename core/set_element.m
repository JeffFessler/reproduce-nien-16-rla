function xout = set_element(xin,y,flag,map)
% x ~= 0, and y = flag

x = embed(xin,map~=0);
x(map==flag) = y;
xout = masker(x,map~=0);
