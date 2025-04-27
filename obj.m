function L = obj(w, y, x)
    m = size(x,2);
    f = zeros(m,1);
    for i = 1:m
        f(i) = sigmoid(w' * x(:,i));
    end
    eps = 1e-8; % protectie log(0)
    L = -(1/m) * sum(y .* log(f + eps) + (1 - y) .* log(1 - f + eps));
end
