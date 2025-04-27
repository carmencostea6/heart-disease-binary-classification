function [precizia, sensibilitatea, acuratetea, F1_score, C] = evaluare_model(Abar_test, e_test, X, x, a, b)
    % Forward propagation pe setul de testare
    Z_test = Abar_test * X;
    H_test = softplusplus(Z_test, a, b);
    y1_test = H_test * x;
    y_test = sigmoid(y1_test);
    predictii_test = double(y_test >= 0.5);% Predictii finale
    C = confusionmat(e_test, predictii_test);   % Matricea de confuzie

    RP = C(2,2); 
    FP = C(1,2); 
    FN = C(2,1); 
    RN = C(1,1); 

    precizia = RP / (RP + FP);
    sensibilitatea = RP / (RP + FN);
    acuratetea = (RP + RN) / (RP + RN + FP + FN);
    F1_score = 2 * (precizia * sensibilitatea) / (precizia + sensibilitatea);

end
