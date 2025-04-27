    %% ALEGERE BAZA DE DATE SI IMPARTIRE DATE(ANTRENARE+TESTARE)
    T = readtable('heart.csv');
    
    % Transformare coloane categorice(stringuri) in coloane numerice
    T.Sex = double(categorical(T.Sex));%F=1,M=2
    T.ChestPainType = double(categorical(T.ChestPainType));%ASY=1,ATA=2,NAP=3
    T.RestingECG = double(categorical(T.RestingECG));%Normal=2,ST=3
    T.ExerciseAngina = double(categorical(T.ExerciseAngina));%N=1,Y=2
    T.ST_Slope = double(categorical(T.ST_Slope));%Flat=2, Up=3
    %disp(T)
    A = T{:,1:end-1}; % input 
    e = T{:,end};     % eticheta (HeartDisease)
    
    [N_total, n] = size(A);%dimensiune originala
    idx = randperm(N_total, 200); % aleg aleatoriu 200 de exemple
    A = A(idx, :); % pastrez doar 200 de exemple
    e = e(idx, :);
    
    N = 200; % nr de exemple cu care lucrez
    N_train = 160;% 80% pt antrenare
    N_test = 40;% 20% pt testare
    
    % Seturi de antrenare
    A_train = A(1:N_train, :);
    e_train = e(1:N_train, :);
    %Seturi de testare
    A_test = A(N_train+1:end, :);
    e_test = e(N_train+1:end, :);
    
    % Ā = [A, 1]
    Abar_train = [A_train, ones(N_train,1)];
    Abar_test = [A_test, ones(N_test,1)];
    
    %% Retea Neuronala
    m=20;%nr neuroni strat ascuns
    a=1;
    b=2;%parametrii pt Soft++
    
    [N_train, n_plus_1] = size(Abar_train); % n+1 caracteristici
    n = n_plus_1 - 1; % caracteristici reale
    
    rng(0); % pentru reproductibilitate
    X = randn(n_plus_1, m) * 0.01; % greutati strat ascuns (Abar -> strat ascuns)
    x = randn(m,1) * 0.01;         % greutati strat ascuns -> iesire
    
    Z=Abar_train*X;% ĀX
    H=softplusplus(Z,a,b);% g(ĀX) - activare strat ascuns  
    y1=H*x;%vector de predictii
    y=sigmoid(y1);%pt clasificare binara vreau ca y sa aiba valori intre 0 si 1
    loss=loss_crossentropy(e_train,y);
    %% METODA GRADIENT
    % Setari
    alfa = 0.003;
    epsilon = 1e-5;
    maxIter = 5000;
    m = 20;
    a = 1;
    b = 2;
    
    % Antrenare cu metoda gradientului
    [X_final, x_final, errors, norms, times] = gradient(Abar_train, e_train, m, a, b, alfa, epsilon, maxIter);
    figure;
    % 1. Eroare vs Iteratii
    subplot(2,2,1); 
    plot(1:length(errors), errors, 'LineWidth', 2);
    xlabel('Iteratii');
    ylabel('Eroare (Loss)');
    title('Eroare vs Iteratii');
    grid on;
    % 2. Eroare vs Timp
    subplot(2,2,2); 
    plot(times, errors, 'LineWidth', 2);
    xlabel('Timp (secunde)');
    ylabel('Eroare (Loss)');
    title('Eroare vs Timp');
    grid on;
    % 3. Norma gradientului vs Iteratii
    subplot(2,2,3); 
    semilogy(1:length(norms), norms, 'LineWidth', 2);
    xlabel('Iteratii');
    ylabel('Norma gradientului');
    title('Norma gradientului vs Iteratii');
    grid on;
    % 4. Norma gradientului vs Timp
    subplot(2,2,4); 
    semilogy(times, norms, 'LineWidth', 2);
    xlabel('Timp (secunde)');
    ylabel('Norma gradientului');
    title('Norma gradientului vs Timp');
    grid on;
    sgtitle('Evolutia antrenarii - Metoda Gradient');
    
    %Performanta sarcinii de invatare
    [precizie_grad, sensibilitate_grad, acuratete_grad, F1_grad, C_grad] = evaluare_model(Abar_test, e_test, X_final, x_final, a, b);
    
    disp('Matricea de confuzie - Gradient normal:');
    disp(C_grad);
    fprintf('Precizie: %.4f\n', precizie_grad);
    fprintf('Sensibilitate: %.4f\n', sensibilitate_grad);
    fprintf('Acuratete: %.4f\n', acuratete_grad);
    fprintf('F1 Score: %.4f\n', F1_grad);
    
    
    %% METODA GRADIENT STOCASTICA
    % Setari
    alfa = 0.003;
    epsilon = 1e-5;
    maxIter = 5000;
    m = 20;
    a = 1;
    b = 2;
    % Antrenare cu metoda gradientului stocastic
    [X_sgd, x_sgd, errors_sgd, norms_sgd, times_sgd] = gradient_stocastic(Abar_train, e_train, m, a, b, alfa, epsilon, maxIter);
    window = 200; % fereastra de mediere-vad mai bine grafic
    errors_sgd = movmean(errors_sgd, window);
    norms_sgd = movmean(norms_sgd, window);
    figure;
    % 1. Eroare vs Iteratii
    subplot(2,2,1); 
    plot(1:length(errors_sgd), errors_sgd, 'LineWidth', 2);
    xlabel('Iteratii');
    ylabel('Eroare (Loss)');
    title('Eroare vs Iteratii - SGD');
    grid on;
    % 2. Eroare vs Timp
    subplot(2,2,2); 
    plot(times_sgd, errors_sgd, 'LineWidth', 2);
    xlabel('Timp (secunde)');
    ylabel('Eroare (Loss)');
    title('Eroare vs Timp - SGD');
    grid on;
    % 3. Norma gradientului vs Iteratii
    subplot(2,2,3); 
    semilogy(1:length(norms_sgd), norms_sgd, 'LineWidth', 2);
    xlabel('Iteratii');
    ylabel('Norma gradientului');
    title('Norma gradientului vs Iteratii - SGD');
    grid on;
    % 4. Norma gradientului vs Timp
    subplot(2,2,4); 
    semilogy(times_sgd, norms_sgd, 'LineWidth', 2);
    xlabel('Timp (secunde)');
    ylabel('Norma gradientului');
    title('Norma gradientului vs Timp - SGD');
    grid on;
    sgtitle('Evolutia antrenarii - Metoda Gradient Stocastic');
    
    %Performanta sarcinii de invatare
    [precizie_sgd, sensibilitate_sgd, acuratete_sgd, F1_sgd, C_sgd] = evaluare_model(Abar_test, e_test, X_sgd, x_sgd, a, b);
    disp('Matricea de confuzie - Gradient Stocastic:');
    disp(C_sgd);
    fprintf('Precizie: %.4f\n', precizie_sgd);
    fprintf('Sensibilitate: %.4f\n', sensibilitate_sgd);
    fprintf('Acuratete: %.4f\n', acuratete_sgd);
    fprintf('F1 Score: %.4f\n', F1_sgd);
    
    %% METODA NEWTON
    epsilon = 1e-5;
    maxIter = 5000;
    % Antrenare Newton
    [w_newton, errors_newton, norms_newton, times_newton] = Newton(Abar_train', e_train, epsilon, maxIter);
    % Ploturi evolutie Newton
    window = 200; % fereastra de mediere-vad mai bine grafic
    errors_newton = movmean(errors_newton, window);
    norms_newton = movmean(norms_newton, window);
    figure;
    % 1. Eroare vs Iteratii
    subplot(2,2,1);
    plot(1:length(errors_newton), errors_newton, 'LineWidth', 2);
    xlabel('Iteratii');
    ylabel('Loss');
    title('Eroare vs Iteratii - Newton');
    grid on;
    % 2. Eroare vs Timp
    subplot(2,2,2);
    plot(times_newton, errors_newton, 'LineWidth', 2);
    xlabel('Timp (secunde)');
    ylabel('Loss');
    title('Eroare vs Timp - Newton');
    grid on;
    % 3. Norma gradientului vs Iteratii
    subplot(2,2,3);
    semilogy(1:length(norms_newton), norms_newton, 'LineWidth', 2);
    xlabel('Iteratii');
    ylabel('Norma gradientului');
    title('Norma gradientului vs Iteratii - Newton');
    grid on;
    % 4. Norma gradientului vs Timp
    subplot(2,2,4);
    semilogy(times_newton, norms_newton, 'LineWidth', 2);
    xlabel('Timp (secunde)');
    ylabel('Norma gradientului');
    title('Norma gradientului vs Timp - Newton');
    grid on;
    sgtitle('Evolutia antrenarii - Metoda Newton');
    
    % Predictii pe setul de testare - metoda Newton
    m_test = size(Abar_test,1);
    functii_test = zeros(m_test,1);
    for i = 1:m_test
        functii_test(i) = sigmoid(w_newton' * Abar_test(i,:)');
    end
    
    predictii_test_newton = double(functii_test >= 0.5);
    
    % Matricea de confuzie si performanta - Newton
    C_newton = confusionmat(e_test, predictii_test_newton);
    
    RP = C_newton(2,2);
    FP = C_newton(1,2);
    FN = C_newton(2,1);
    RN = C_newton(1,1);
    
    precizie_newton = RP / (RP + FP);
    sensibilitate_newton = RP / (RP + FN);
    acuratete_newton = (RP + RN) / (RP + RN + FP + FN);
    F1_newton = 2 * (precizie_newton * sensibilitate_newton) / (precizie_newton + sensibilitate_newton);
    % Afisare rezultate
    disp('Matricea de confuzie - Newton:');
    disp(C_newton);
    fprintf('Precizie: %.4f\n', precizie_newton);
    fprintf('Sensibilitate: %.4f\n', sensibilitate_newton);
    fprintf('Acuratete: %.4f\n', acuratete_newton);
    fprintf('F1 Score: %.4f\n', F1_newton);
    %% Comparare Metode
    % Comparatie Performante- Acuratete si F Score
    metode = {'Gradient', 'SGD', 'Newton'};
    acuratete = [acuratete_grad, acuratete_sgd, acuratete_newton] * 100; 
    F1_scores = [F1_grad, F1_sgd, F1_newton] * 100; 
    
    figure;
    bar([acuratete; F1_scores]');
    ylabel('Procent (%)');
    title('Comparatie Acuratete si F Score');
    legend('Acuratete', 'F Score', 'Location', 'northwest');
    set(gca, 'XTickLabel', metode);
    grid on;
    
    % Comparatie metode - Timp si Numar Iteratii
    timpuri = [times(end), times_sgd(end), times_newton(end)];
    timpuri=timpuri*1000;% scalez pt a vedea grafic
    numar_iteratii = [length(errors), length(errors_sgd), length(errors_newton)];
    figure;
    bar([timpuri; numar_iteratii]');
    ylabel('Timp (secunde) / Numar Iteratii');
    title('Comparatie Timp Antrenare si Numar Iteratii');
    legend('Timp', 'Numar Iteratii', 'Location', 'northwest');
    set(gca, 'XTickLabel', metode);
    grid on;

