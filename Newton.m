function [w_final, errors, norms, times] = Newton(x, y, epsilon, maxIter)
    [n, N] = size(x);
    rng(0); % pentru reproductibilitate
    w = randn(n,1) * 0.01; % Initializare
    errors = zeros(1, maxIter);
    norms = zeros(1, maxIter);
    times = zeros(1, maxIter);

    iter = 0;
    tic;

    while norm(Grad(w, y, x)) > epsilon && iter < maxIter
        iter = iter + 1;

        g = Grad(w, y, x); % Gradient
        lambda = 1e-4; % regularizare mica
        H = hessiana(w, y, x);
        H = H + lambda * eye(size(H)); % regularizare
        w = w - H \ g'; % pas Newton

       
        errors(iter) = obj(w, y, x); 
        norms(iter) = norm(g);
        times(iter) = toc;
    end

    errors = errors(1:iter);
    norms = norms(1:iter);
    times = times(1:iter);
    w_final = w;
end
