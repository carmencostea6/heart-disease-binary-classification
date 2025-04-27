function g=softplusplus(z,a,b)
g=log(1+exp(a.*z))+(z./b)-log(2);
end