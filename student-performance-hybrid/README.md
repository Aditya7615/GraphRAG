# Hybrid Fuzzy-Neural Network for Student Performance Prediction

A hybrid intelligent system combining **fuzzy logic** (rule-based expert knowledge) with **neural networks** (data-driven learning) to predict student performance levels.

---

## Overview

This system predicts student performance (Poor, Average, Good) based on three input metrics using a hybrid approach:

- **Fuzzy Logic**: Handles linguistic/uncertain inputs through membership functions and IF-THEN rules
- **Neural Network**: Learns patterns from data to refine and adjust fuzzy outputs

## Inputs

| Input | Range | Membership Functions |
|-------|-------|---------------------|
| **Attendance** | 0-100% | Low, Medium, High |
| **Assignment Marks** | 0-100% | Poor, Average, Good |
| **Test Marks** | 0-100% | Poor, Average, Good |

## Output

| Output | Range | Classes |
|--------|-------|---------|
| **Performance Level** | 0-1 | Poor (0-0.33), Average (0.33-0.66), Good (0.66-1) |

---

## How Fuzzy Logic and Neural Networks Are Integrated

### Architecture: ANFIS-Inspired Hybrid

```
                    ┌──────────────────┐
                    │   CRISP INPUTS   │
                    │ Attendance (X1) │
                    │ Assignment (X2) │
                    │ Test Marks (X3)  │
                    └────────┬─────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                 LAYER 1: FUZZIFICATION                      │
│  Converts crisp inputs to fuzzy membership degrees         │
│                                                            │
│  μ_Low(X1), μ_Medium(X1), μ_High(X1)                       │
│  μ_Poor(X2), μ_Average(X2), μ_Good(X2)                      │
│  μ_Poor(X3), μ_Average(X3), μ_Good(X3)                      │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│                    LAYER 2: FUZZY RULES                     │
│  Applies expert-defined IF-THEN rules                      │
│                                                            │
│  Rule 1: IF Attendance=Low AND Assign=Poor AND Test=Poor   │
│          THEN Performance=Poor                             │
│  Rule 2: IF Attendance=Medium AND ... THEN Performance=Avg │
│  ... (9 rules total)                                       │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│               LAYER 3: DEFUZZICATION                       │
│  Converts fuzzy output to crisp value using centroid       │
│  Output: Initial performance prediction (0-1)              │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│            LAYER 4: NEURAL NETWORK REFINEMENT              │
│  Trained feedforward network adjusts fuzzy output          │
│                                                            │
│  Input: Fuzzy output (0-1)                                 │
│  Hidden: 10 tansig neurons                                 │
│  Output: Refinement factor                                 │
│                                                            │
│  Final = Fuzzy_Output + α × Neural_Adjustment              │
│  (α = combination weight, typically 0.3)                    │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ PERFORMANCE      │
                    │ PREDICTION       │
                    │ (0-1 + refinement)│
                    └──────────────────┘
```

### Integration Method

The hybrid system uses a **sequential integration** approach:

1. **Stage 1 - Fuzzy Logic (Symbolic Processing)**
   - Expert knowledge encoded as linguistic rules
   - Handles uncertainty in input measurements
   - Provides interpretable, explainable predictions

2. **Stage 2 - Neural Network (Numerical Learning)**
   - Learns from fuzzy system outputs
   - Captures subtle patterns the rules may miss
   - Refines predictions based on training data

### Why This Integration Works

| Aspect | Fuzzy Logic | Neural Network |
|--------|-------------|----------------|
| **Knowledge** | Rule-based, explicit | Learned, implicit |
| **Uncertainty** | Handles naturally | Approximates |
| **Interpretability** | High (IF-THEN rules) | Low (black box) |
| **Learning** | Manual rule definition | Automatic from data |
| **Adaptability** | Static rules | Dynamic learning |

**Combined Benefits:**
- Fuzzy logic provides interpretable rules and handles linguistic uncertainty
- Neural network learns patterns and adjusts rule outputs for better accuracy
- The system can be improved by retraining with new student data

---

## Fuzzy Rules (9 Rules)

| Rule | Attendance | Assignment | Test | Performance |
|------|------------|------------|------|-------------|
| 1 | Low | Poor | Poor | Poor |
| 2 | Low | Average | Average | Poor |
| 3 | Low | Good | Good | Average |
| 4 | Medium | Poor | Poor | Poor |
| 5 | Medium | Average | Average | Average |
| 6 | Medium | Good | Good | Good |
| 7 | High | Poor | Poor | Average |
| 8 | High | Average | Average | Average |
| 9 | High | Good | Good | Good |

### Rule Logic Explanation

- **Rule 1-3 (Low Attendance)**: Even good marks can't fully compensate for poor attendance
- **Rule 4-6 (Medium Attendance)**: Balanced performance gets balanced output
- **Rule 7-9 (High Attendance)**: Attendance shows engagement, but marks still matter

---

## Membership Functions

### Attendance (Input 1)

| Function | Type | Parameters |
|----------|------|------------|
| Low | Trapezoidal | [0, 0, 30, 50] |
| Medium | Triangular | [30, 55, 80] |
| High | Trapezoidal | [60, 80, 100, 100] |

### Assignment Marks (Input 2)

| Function | Type | Parameters |
|----------|------|------------|
| Poor | Trapezoidal | [0, 0, 25, 40] |
| Average | Triangular | [30, 50, 70] |
| Good | Trapezoidal | [60, 75, 100, 100] |

### Test Marks (Input 3)

| Function | Type | Parameters |
|----------|------|------------|
| Poor | Trapezoidal | [0, 0, 25, 40] |
| Average | Triangular | [30, 50, 70] |
| Good | Trapezoidal | [60, 75, 100, 100] |

### Performance Level (Output)

| Function | Type | Parameters |
|----------|------|------------|
| Poor | Trapezoidal | [0, 0, 0.2, 0.4] |
| Average | Triangular | [0.25, 0.5, 0.75] |
| Good | Trapezoidal | [0.6, 0.8, 1, 1] |

---

## Neural Network Architecture

```
Input:  1 neuron  (fuzzy output value)
Hidden: 10 neurons (tansig activation)
Output: 1 neuron  (refinement adjustment)

Training:
- Algorithm: Levenberg-Marquardt (trainlm)
- Goal: MSE < 0.001
- Max Epochs: 100
- Learning Rate: 0.01
```

The neural network is trained on data generated from the fuzzy system, learning to adjust/refine the fuzzy outputs for better accuracy.

---

## Files

```
student-performance-hybrid/
├── studentPerformanceHybrid.m    # Main MATLAB/Octave script
├── studentPerformanceFuzzy.fis   # Saved fuzzy inference system
├── neuralNetwork.mat            # Trained neural network
└── README.md                    # This file
```

---

## How to Run

### Prerequisites

- **MATLAB** (R2018b+) with Fuzzy Logic Toolbox and Neural Network Toolbox
- **GNU Octave** (free alternative) with fuzzy-logic-toolkit package

### In MATLAB

```matlab
cd student-performance-hybrid
studentPerformanceHybrid
```

### In Octave

```octave
cd student-performance-hybrid
studentPerformanceHybrid
```

Note: Octave may require `fuzzy-logic-toolkit` package. Install with:
```octave
pkg install fuzzy-logic-toolkit
```

---

## Sample Output

```
==================================================
TEST RESULTS
==================================================

Attendance          Assignment         Test Marks          Fuzzy Only       Hybrid
----------          ----------         ---------          ---------       ------
85                  90                 88                 0.850          0.892
20                  30                 25                 0.150          0.198
50                  55                 50                 0.500          0.501
90                  20                 30                 0.350          0.389
30                  85                 90                 0.750          0.720
70                  65                 70                 0.650          0.648
40                  45                 40                 0.350          0.361
95                  95                 92                 0.950          0.945
15                  15                 20                 0.100          0.127

Performance Classification:
Attendance          Assignment         Test Marks          Classification
----------          ----------         ---------          --------------
85                  90                 88                 Good
20                  30                 25                 Poor
50                  55                 50                 Average
90                  20                 30                 Average
30                  85                 90                 Average
70                  65                 70                 Average
40                  45                 40                 Average
95                  95                 92                 Good
15                  15                 20                 Poor
```

---

## Visualization

The script generates 9 plots:

1. **Attendance Membership Functions** - Low/Medium/High curves
2. **Assignment Membership Functions** - Poor/Average/Good curves
3. **Test Marks Membership Functions** - Poor/Average/Good curves
4. **Performance Output Membership** - Poor/Average/Good output curves
5. **Neural Network Training Progress** - MSE vs Epoch
6. **Fuzzy vs Hybrid Output Comparison** - Side-by-side bar chart
7. **3D Surface Plot** - Performance vs Attendance & Assignment
8. **Output Distribution** - Histogram of fuzzy vs hybrid outputs
9. **System Architecture** - Diagram of hybrid system flow

---

## Customization

### Add Your Own Rules

Edit the `ruleList` matrix in the script:

```matlab
% Format: [Attendance, Assignment, Test, Performance, Weight, Connection]
% Performance: 1=Poor, 2=Average, 3=Good
ruleList = [
    1 1 1 1 1 1;  % Your rule here
    % ...
];
fis = addrule(fis, ruleList);
```

### Adjust Neural Weight

The combination weight `α` controls neural network influence:

```matlab
hybridOutput = fuzzyOutput + 0.3 * neuralAdjustment;
```

Change `0.3` to:
- **Lower (0.1-0.2)**: Trust fuzzy rules more
- **Higher (0.4-0.5)**: Trust neural network more

### Retrain with New Data

```matlab
% Load your custom data
X = [attendance_data, assignment_data, test_data];
Y = actual_performance_data; % 0-1 range

% Retrain network
net = train(net, X', Y');
save('neuralNetwork.mat', 'net');
```

---

## Theory

### Why Hybrid Fuzzy-Neural?

1. **Fuzzy Logic Alone**
   - Pros: Interpretable, explainable rules
   - Cons: Static rules, doesn't learn from data

2. **Neural Network Alone**
   - Pros: Learns patterns, adaptive
   - Cons: Black box, requires lots of data

3. **Hybrid (This System)**
   - Combines interpretability with learning
   - Fewer training samples needed (fuzzy rules guide initial behavior)
   - More robust to edge cases

### ANFIS Connection

This system is inspired by **ANFIS (Adaptive Neuro-Fuzzy Inference System)**:

| ANFIS Layer | Our System Component |
|-------------|---------------------|
| Fuzzification | Membership functions |
| Rule evaluation | Fuzzy rules (IF-THEN) |
| Normalization | Weighted rule combination |
| Defuzzification | Centroid calculation |
| Output | Neural refinement layer |

---

## License

MIT License

---

## Author

Created as a demonstration of hybrid intelligent systems combining symbolic (fuzzy logic) and connectionist (neural network) approaches.