function L=loss_crossentropy(e,y)
N=length(e);
eps=1e-8;%eroare pt log(0)
suma=0;
for i=1:N
    suma=suma+(e(i)*log(y(i)+eps)+(1-e(i))*log(1-y(i)+eps));
end
L=(-1/N)*suma;
end