%% METODA GRADIENT
function [X, x, errors, norms, times] = gradient(Abar_train, e_train, m, a, b, alfa, epsilon, maxIter)
    [N_train, n_plus_1] = size(Abar_train);
    rng(0); % pentru reproductibilitate
    X = randn(n_plus_1, m) * 0.01;
    x = randn(m, 1) * 0.01;

    errors = zeros(1, maxIter);
    norms = zeros(1, maxIter);
    times = zeros(1, maxIter);

    iter = 0;
    f0 = 1;
    f1 = 0;
    norma_gradient = inf;

    tic; 

    while abs(f1 - f0) >= epsilon && iter < maxIter
        iter = iter + 1;
        f0 = f1;
        Z = Abar_train * X; 
        H = softplusplus(Z, a, b);
        y1 = H * x; 
        y = sigmoid(y1);
        f1 = loss_crossentropy(e_train, y); 
        delta = y - e_train;%derivata fct. de pierdere combinata cu sigmoid
        % Gradient fata de x
        grad_x = H' * delta / N_train;
        % Gradient fata de X
        dH = (delta * x') / N_train;
        dSoft = d_softplusplus(Z, a, b);
        grad_X = Abar_train' * (dH .* dSoft);
        norma_gradient = sqrt(sum(grad_x.^2) + sum(grad_X(:).^2));%norma gradientului
        
        % Metoda Gradient
        x = x - alfa * grad_x;
        X = X - alfa * grad_X;

        errors(iter) = f1;
        norms(iter) = norma_gradient;
        times(iter) = toc; % timpul curent la fiecare iteratie
    end
    % Trunchiere vectorii la numarul real de iteratii
    errors = errors(1:iter);
    norms = norms(1:iter);
    times = times(1:iter);

end
