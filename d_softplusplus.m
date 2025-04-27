function g_prime = d_softplusplus(z, a, b)
    g_prime = (a * exp(a*z)) ./ (1 + exp(a*z)) + 1/b;
end
