% Output g este o valoare intre 0 si 1
function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end