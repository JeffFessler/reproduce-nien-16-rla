function y = get_element(x,flag,map)
% x ~= 0, and y = flag

y = masker(embed(x,map~=0),map==flag);

