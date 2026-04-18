% Hybrid Fuzzy-Neural Network for Student Performance Prediction
% Combines fuzzy logic (rule-based) with neural network (learning) approaches
%
% Inputs: Attendance (%), Assignment Marks (%), Test Marks (%)
% Output: Performance Level (Poor: 0-0.33, Average: 0.33-0.66, Good: 0.66-1.0)

%% Clean workspace
clear all
close all
clc

%% ============================================================
% PART 1: FUZZY LOGIC SUBSYSTEM
% ============================================================

fprintf('==================================================\n');
fprintf('HYBRID FUZZY-NEURAL NETWORK STUDENT PERFORMANCE\n');
fprintf('==================================================\n\n');

% Create Fuzzy Inference System
fis = newfis('StudentPerformance');

% ===== INPUT 1: Attendance (0-100%) =====
fis = addvar(fis, 'input', 'Attendance', [0 100]);

% Low Attendance - triangular
fis = addmf(fis, 'input', 1, 'Low', 'trapmf', [0 0 30 50]);

% Medium Attendance - triangular
fis = addmf(fis, 'input', 1, 'Medium', 'trimf', [30 55 80]);

% High Attendance - trapezoidal
fis = addmf(fis, 'input', 1, 'High', 'trapmf', [60 80 100 100]);

% ===== INPUT 2: Assignment Marks (0-100%) =====
fis = addvar(fis, 'input', 'AssignmentMarks', [0 100]);

% Poor Assignment - trapezoidal
fis = addmf(fis, 'input', 2, 'Poor', 'trapmf', [0 0 25 40]);

% Average Assignment - triangular
fis = addmf(fis, 'input', 2, 'Average', 'trimf', [30 50 70]);

% Good Assignment - trapezoidal
fis = addmf(fis, 'input', 2, 'Good', 'trapmf', [60 75 100 100]);

% ===== INPUT 3: Test Marks (0-100%) =====
fis = addvar(fis, 'input', 'TestMarks', [0 100]);

% Poor Test - trapezoidal
fis = addmf(fis, 'input', 3, 'Poor', 'trapmf', [0 0 25 40]);

% Average Test - triangular
fis = addmf(fis, 'input', 3, 'Average', 'trimf', [30 50 70]);

% Good Test - trapezoidal
fis = addmf(fis, 'input', 3, 'Good', 'trapmf', [60 75 100 100]);

% ===== OUTPUT: Performance Level (0-1) =====
fis = addvar(fis, 'output', 'PerformanceLevel', [0 1]);

% Poor Performance
fis = addmf(fis, 'output', 1, 'Poor', 'trapmf', [0 0 0.2 0.4]);

% Average Performance
fis = addmf(fis, 'output', 1, 'Average', 'trimf', [0.25 0.5 0.75]);

% Good Performance
fis = addmf(fis, 'output', 1, 'Good', 'trapmf', [0.6 0.8 1 1]);

% ===== FUZZY RULES =====
% Format: [Attendance, Assignment, Test] -> [Performance, Weight]
% Performance levels: 1=Poor, 2=Average, 3=Good
% ============================================================
% 9 Core fuzzy rules (combining all inputs with performance)
% ============================================================

ruleList = [
    % IF Attendance=Low AND Assignments=Poor AND Tests=Poor THEN Performance=Poor
    1 1 1 1 1 1;

    % IF Attendance=Low AND Assignments=Average AND Tests=Average THEN Performance=Poor
    1 2 2 1 1 1;

    % IF Attendance=Low AND Assignments=Good AND Tests=Good THEN Performance=Average
    1 3 3 2 1 1;

    % IF Attendance=Medium AND Assignments=Poor AND Tests=Poor THEN Performance=Poor
    2 1 1 1 1 1;

    % IF Attendance=Medium AND Assignments=Average AND Tests=Average THEN Performance=Average
    2 2 2 2 1 1;

    % IF Attendance=Medium AND Assignments=Good AND Tests=Good THEN Performance=Good
    2 3 3 3 1 1;

    % IF Attendance=High AND Assignments=Poor AND Tests=Poor THEN Performance=Average
    3 1 1 2 1 1;

    % IF Attendance=High AND Assignments=Average AND Tests=Average THEN Performance=Average
    3 2 2 2 1 1;

    % IF Attendance=High AND Assignments=Good AND Tests=Good THEN Performance=Good
    3 3 3 3 1 1
];

fis = addrule(fis, ruleList);

fprintf('PART 1: FUZZY LOGIC SYSTEM\n');
fprintf('--------------------------\n');
fprintf('Created Mamdani FIS with 3 inputs and 9 rules\n');
fprintf('Inputs: Attendance, Assignment Marks, Test Marks\n');
fprintf('Output: Performance Level (Poor/Average/Good)\n\n');

%% ============================================================
% PART 2: NEURAL NETWORK SUBSYSTEM
% ============================================================
% The neural network learns to adjust fuzzy outputs based on training data
% This creates an ANFIS-like (Adaptive Neuro-Fuzzy Inference System)
% ============================================================

fprintf('PART 2: NEURAL NETWORK TRAINING\n');
fprintf('------------------------------\n');

% Generate training data from fuzzy system
% Creating comprehensive dataset
numSamples = 200;
trainingData = zeros(numSamples, 4);

for i = 1:numSamples
    attendance = rand * 100;
    assignment = rand * 100;
    test = rand * 100;

    % Get fuzzy output
    fuzzyOutput = evalfis([attendance assignment test], fis);

    trainingData(i, :) = [attendance assignment test fuzzyOutput];
end

% Split into features and targets
X = trainingData(:, 1:3);  % Inputs
Y = trainingData(:, 4);    % Fuzzy output

% Create Neural Network
hiddenLayerSize = 10;
net = feedforwardnet(hiddenLayerSize);

% Configure network
net.layers{1}.transferFcn = 'tansig';  % Hyperbolic tangent sigmoid
net.layers{2}.transferFcn = 'purelin'; % Linear output

% Train network using Levenberg-Marquardt
net.trainFcn = 'trainlm';
net.trainParam.epochs = 100;
net.trainParam.lr = 0.01;
net.trainParam.goal = 0.001;

% Prepare data for training
X = con2seq(X');
Y = con2seq(Y');

% Train the network
[net, tr] = train(net, X, Y);

fprintf('Neural Network trained successfully!\n');
fprintf('Training Epochs: %d\n', tr.num_epochs);
fprintf('Final MSE: %.6f\n', tr.perf(tr.num_epochs));
fprintf('Hidden Layer Size: %d neurons\n\n', hiddenLayerSize);

%% ============================================================
% PART 3: HYBRID INTEGRATION
% ============================================================
% The hybrid system combines fuzzy logic (symbolic/rule-based)
% with neural network (numerical/learning-based) in two ways:
%
% 1. FUZZY PREPROCESSING: Fuzzy logic fuzzifies inputs and
%    applies expert rules
%
% 2. NEURAL REFINEMENT: Neural network learns to adjust/
%    refine fuzzy outputs based on real-world data patterns
% ============================================================

fprintf('==================================================\n');
fprintf('PART 3: HYBRID SYSTEM INTEGRATION\n');
fprintf('==================================================\n');
fprintf('\nIntegration Architecture:\n');
fprintf('--------------------------\n');
fprintf('1. FUZZY LAYER: Fuzzifies inputs using membership functions\n');
fprintf('2. RULE LAYER: Applies 9 fuzzy rules for initial inference\n');
fprintf('3. DEFUZZY LAYER: Converts fuzzy output to crisp value\n');
fprintf('4. NEURAL LAYER: Refines output using trained neural network\n');
fprintf('5. OUTPUT: Final performance prediction\n\n');

%% ============================================================
% PART 4: EVALUATION & COMPARISON
% ============================================================

fprintf('==================================================\n');
fprintf('TEST RESULTS\n');
fprintf('==================================================\n');
fprintf('\n%-20s %-15s %-15s %-15s %-15s\n', 'Attendance', 'Assignment', 'Test', 'Fuzzy Only', 'Hybrid');
fprintf('%-20s %-15s %-15s %-15s %-15s\n', '----------', '----------', '-----', '---------', '------');

% Test cases
testCases = [
    85  90  88;   % High performer
    20  30  25;   % Low performer
    50  55  50;   % Average performer
    90  20  30;   % High attendance, poor marks
    30  85  90;   % Low attendance, excellent marks
    70  65  70;   % Good consistent performance
    40  45  40;   % Below average
    95  95  92;   % Excellent across all
    15  15  20;   % Very poor
];

fuzzyOutputs = zeros(size(testCases, 1), 1);
hybridOutputs = zeros(size(testCases, 1), 1);

for i = 1:size(testCases, 1)
    attendance = testCases(i, 1);
    assignment = testCases(i, 2);
    test = testCases(i, 3);

    % Pure fuzzy output
    fuzzyOutputs(i) = evalfis([attendance assignment test], fis);

    % Hybrid: Neural refinement of fuzzy output
    fuzzyVal = evalfis([attendance assignment test], fis);
    neuralAdjustment = net(fuzzyVal);
    hybridOutputs(i) = fuzzyVal + 0.3 * neuralAdjustment; % Weighted combination
    hybridOutputs(i) = min(1, max(0, hybridOutputs(i)));   % Clamp to [0,1]
end

for i = 1:size(testCases, 1)
    fprintf('%-20d %-15d %-15d %-15.3f %-15.3f\n', ...
        testCases(i, 1), testCases(i, 2), testCases(i, 3), ...
        fuzzyOutputs(i), hybridOutputs(i));
end

%% ============================================================
% PART 5: PERFORMANCE CLASSIFICATION
% ============================================================

fprintf('\n\nPerformance Classification:\n');
fprintf('--------------------------\n');
fprintf('%-20s %-15s %-15s %-15s\n', 'Attendance', 'Assignment', 'Test', 'Classification');
fprintf('%-20s %-15s %-15s %-15s\n', '----------', '----------', '-----', '--------------');

for i = 1:size(testCases, 1)
    attendance = testCases(i, 1);
    assignment = testCases(i, 2);
    test = testCases(i, 3);
    hybridOut = hybridOutputs(i);

    if hybridOut < 0.33
        classification = 'Poor';
    elseif hybridOut < 0.66
        classification = 'Average';
    else
        classification = 'Good';
    end

    fprintf('%-20d %-15d %-15d %-15s\n', ...
        attendance, assignment, test, classification);
end

%% ============================================================
% PART 6: VISUALIZATION
% ============================================================

figure('Name', 'Hybrid System Analysis', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);

% 1. Fuzzy Membership Functions
subplot(3, 3, 1);
plotmf(fis, 'input', 1);
title('Attendance Membership');
xlabel('Attendance (%)');
ylabel('Membership');
grid on;

subplot(3, 3, 2);
plotmf(fis, 'input', 2);
title('Assignment Membership');
xlabel('Assignment Marks (%)');
ylabel('Membership');
grid on;

subplot(3, 3, 3);
plotmf(fis, 'input', 3);
title('Test Marks Membership');
xlabel('Test Marks (%)');
ylabel('Membership');
grid on;

% 2. Output Membership
subplot(3, 3, 4);
plotmf(fis, 'output', 1);
title('Performance Level Output');
xlabel('Performance (0-1)');
ylabel('Membership');
grid on;

% 3. Neural Network Error
subplot(3, 3, 5);
plot(tr.perf);
title('Neural Network Training Progress');
xlabel('Epoch');
ylabel('MSE');
grid on;

% 4. Fuzzy vs Hybrid Comparison
subplot(3, 3, 6);
x_vals = 1:size(testCases, 1);
plot(x_vals, fuzzyOutputs, 'b-o', x_vals, hybridOutputs, 'r-s');
title('Fuzzy vs Hybrid Output');
xlabel('Test Case');
ylabel('Performance Level');
legend('Fuzzy', 'Hybrid');
grid on;

% 5. 3D Surface - Fuzzy System
subplot(3, 3, 7);
[X, Y] = meshgrid(0:10:100, 0:10:100);
Z = zeros(size(X));
for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        Z(i, j) = evalfis([X(i,j) Y(i,j) 50], fis); % Test Marks = 50
    end
end
surf(X, Y, Z, 'EdgeColor', 'interp');
title('Fuzzy Surface (Test=50%)');
xlabel('Attendance (%)');
ylabel('Assignment (%)');
zlabel('Performance');
colorbar;

% 6. Performance Distribution
subplot(3, 3, 8);
histogram(fuzzyOutputs, 10, 'FaceColor', 'b', 'Alpha', 0.7);
hold on;
histogram(hybridOutputs, 10, 'FaceColor', 'r', 'Alpha', 0.7);
title('Output Distribution');
xlabel('Performance Level');
ylabel('Frequency');
legend('Fuzzy', 'Hybrid');
grid on;

% 7. System Architecture
subplot(3, 3, 9);
axis off;
arch_text = {
    'HYBRID SYSTEM ARCHITECTURE'
    ''
    '┌─────────────────────────────────┐'
    '│      CRISP INPUTS               │'
    '│  Attendance, Assignment, Test   │'
    '└─────────────┬───────────────────┘'
    '              │'
    '              ▼'
    '┌─────────────────────────────────┐'
    '│  FUZZY LAYER (Fuzzification)    │'
    '│  • Attendance MFs (3)          │'
    '│  • Assignment MFs (3)          │'
    '│  • Test MFs (3)                │'
    '└─────────────┬───────────────────┘'
    '              │'
    '              ▼'
    '┌─────────────────────────────────┐'
    '│  RULE LAYER (Inference)         │'
    '│  9 Fuzzy Rules (IF-THEN)        │'
    '│  Mamdani MIN-MAX Logic          │'
    '└─────────────┬───────────────────┘'
    '              │'
    '              ▼'
    '┌─────────────────────────────────┐'
    '│  DEFUZZIFICATION (Centroid)     │'
    '│  Crisp fuzzy output (0-1)        │'
    '└─────────────┬───────────────────┘'
    '              │'
    '              ▼'
    '┌─────────────────────────────────┐'
    '│  NEURAL LAYER (Refinement)       │'
    '│  10 neurons (tansig)            │'
    '│  Learns patterns from data       │'
    '└─────────────┬───────────────────┘'
    '              │'
    '              ▼'
    '┌─────────────────────────────────┐'
    '│  FINAL OUTPUT                   │'
    '│  Performance Level (0-1)         │'
    '│  Poor/Average/Good              │'
    '└─────────────────────────────────┘'
};
text(0.1, 0.5, arch_text, 'FontName', 'Courier', 'FontSize', 8, 'VerticalAlignment', 'middle');

%% ============================================================
% SAVE SYSTEMS
% ============================================================

writefis(fis, 'studentPerformanceFuzzy.fis');
save('neuralNetwork.mat', 'net');

fprintf('\n==================================================\n');
fprintf('FILES SAVED\n');
fprintf('==================================================\n');
fprintf('1. studentPerformanceFuzzy.fis - Fuzzy inference system\n');
fprintf('2. neuralNetwork.mat - Trained neural network\n');
fprintf('\n==================================================\n');
fprintf('READY FOR USE\n');
fprintf('==================================================\n');

%% ============================================================
% HELPER FUNCTION: Quick Prediction
% ============================================================
% function performance = predictPerformance(attendance, assignment, test, fis, net)
%     fuzzy_out = evalfis([attendance assignment test], fis);
%     neural_adj = net(fuzzy_out);
%     performance = min(1, max(0, fuzzy_out + 0.3 * neural_adj));
% end
% ============================================================