%% METODA GRADIENT STOCASTIC
function [X, x, errors, norms, times] = gradient_stocastic(Abar_train, e_train, m, a, b, alfa, epsilon, maxIter)
    [N_train, n_plus_1] = size(Abar_train);
    rng(0); % reproducibilitate
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
        % Aleg un exemplu aleatoriu
        i = randi(N_train); % indice aleator intre 1 si N_train
        Abar_i = Abar_train(i, :); % exemplul selectat
        e_i = e_train(i); % eticheta asociata
        Z_i = Abar_i * X; 
        H_i = softplusplus(Z_i, a, b); 
        y1_i = H_i * x;
        y_i = sigmoid(y1_i);
        % Loss doar pentru exemplul i (optional, pentru verificare)
        % f1_i = loss_crossentropy(e_i, y_i);
        delta_i = y_i - e_i; % derivata combinata pierdere+sigmoid

        % Gradient fata de x si X
        grad_x = H_i' * delta_i;
        dH_i = delta_i * x';
        dSoft_i = d_softplusplus(Z_i, a, b);
        grad_X = Abar_i' * (dH_i .* dSoft_i);
        norma_gradient = sqrt(sum(grad_x.^2) + sum(grad_X(:).^2));

        % Metoda Gradient
        x = x - alfa * grad_x;
        X = X - alfa * grad_X;

        % Evaluez loss pe toate datele (ca sa urmaresc evolutia corecta)
        Z_full = Abar_train * X;
        H_full = softplusplus(Z_full, a, b);
        y1_full = H_full * x;
        y_full = sigmoid(y1_full);
        f1 = loss_crossentropy(e_train, y_full);

        errors(iter) = f1;
        norms(iter) = norma_gradient;
        times(iter) = toc;
    end
    % Trunchiere vectorii la numarul real de iteratii
    errors = errors(1:iter);
    norms = norms(1:iter);
    times = times(1:iter);
end
