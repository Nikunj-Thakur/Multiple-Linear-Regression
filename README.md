# 🏠 Multiple Linear Regression: House Price Prediction

> Predicting house prices using multivariate linear regression with gradient descent optimization

![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/MachineLearning-Multiple%20Regression-brightgreen?style=flat-square)

---

## 📌 Overview

This project implements **multiple linear regression from scratch** to predict house prices based on multiple features. The model uses **gradient descent optimization** to find the optimal parameters (weights and bias) that minimize the cost function.

### 🎯 Key Objectives

✅ **Multiple Feature Analysis**: Incorporate multiple input features for prediction  
✅ **Gradient Descent Optimization**: Iterative parameter optimization  
✅ **Cost Function Minimization**: Monitor convergence and model performance  
✅ **Vectorized Operations**: Efficient computation using NumPy  

---

## 🤔 Problem Statement

**Goal**: Predict house sale prices based on multiple property characteristics

### Features (Input Variables)
- **Square Footage (X₁)**: Size of the house in square feet
- **Number of Bedrooms (X₂)**: Count of bedrooms
- **Number of Floors (X₃)**: Count of floors/stories
- **Age of Property (X₄)**: Age of the house in years

### Target Variable
- **Sale Price (y)**: House price in thousands of dollars

### The Mathematical Model
```
Price = w₁(Square Footage) + w₂(Bedrooms) + w₃(Floors) + w₄(Age) + b
```

Where:
- **w** = weights/coefficients for each feature
- **b** = bias/intercept term

---

## 📂 Project Structure

```
Multiple Linear Regression/
├── house_price_prediction_model.py    # Main training and prediction script
├── model_parameters.py                # Model functions and utilities
└── README.md                          # This file
```

---

## 📄 Files Description

### `house_price_prediction_model.py`
The main execution script that:
- Defines the training dataset with 3 sample houses
- Initializes model parameters
- Demonstrates both looping and vectorized prediction methods
- Computes cost and gradients
- Runs gradient descent optimization
- Visualizes cost function convergence

**Key Variables**:
- `X_train`: Feature matrix (3 samples × 4 features)
- `y_train`: Target prices
- `alpha`: Learning rate (step size for gradient descent)
- `iterations`: Number of optimization steps

### `model_parameters.py`
Contains all model functions:

| Function | Purpose |
|----------|---------|
| `predict_simple_loop()` | Prediction using explicit loops |
| `predict_vectorized()` | Prediction using NumPy vector operations |
| `compute_cost()` | Calculates Mean Squared Error (MSE) |
| `compute_gradient()` | Computes gradients for optimization |
| `gradient_descent()` | Main optimization algorithm |

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.7+
NumPy
Matplotlib
```

### Installation
```bash
pip install numpy matplotlib
```

### Running the Model
```bash
python house_price_prediction_model.py
```

### Expected Output
- Predictions from both loop and vectorized methods
- Cost function value
- Gradients with respect to weights and bias
- Trained parameters (w_final, b_final)
- Cost convergence plot

---

## 📊 Training Data

```python
X_train = [
    [2104,  5, 1, 45],    # House 1: 2104 sqft, 5 bed, 1 floor, 45 yrs
    [1416,  3, 2, 40],    # House 2: 1416 sqft, 3 bed, 2 floor, 40 yrs
    [852,   2, 1, 35]     # House 3: 852 sqft, 2 bed, 1 floor, 35 yrs
]
y_train = [460, 232, 178]  # Prices in thousands
```

---

## 🔬 Algorithm Details

### 1. **Prediction**
```
ŷ = w·x + b = Σ(wᵢ × xᵢ) + b
```
Uses dot product for efficient computation.

### 2. **Cost Function (MSE)**
```
J(w,b) = (1/2m) × Σ(ŷ - y)²
```
Where m = number of training examples

### 3. **Gradients**
```
∂J/∂w = (1/m) × Σ(ŷ - y) × x
∂J/∂b = (1/m) × Σ(ŷ - y)
```

### 4. **Gradient Descent Update**
```
w := w - α × ∂J/∂w
b := b - α × ∂J/∂b
```
Where α = learning rate

---

## 📈 Hyperparameters

Current configuration in the script:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` (Learning Rate) | 5.0e-7 | Controls step size in gradient descent |
| `iterations` | 1000 | Number of optimization steps |
| `initial_w` | [0, 0, 0, 0] | Initial weights |
| `initial_b` | 0 | Initial bias |

**Note**: The learning rate is small to prevent overshooting during optimization.

---

## 📊 Visualization

The script generates a cost convergence plot with two subplots:
1. **Full History**: Cost vs iteration (all 1000 iterations)
2. **Tail View**: Cost vs iteration (last 900 iterations) for detailed analysis

This helps identify:
- Whether the algorithm is converging
- Optimal number of iterations
- Learning rate appropriateness

---

## 💡 Key Concepts

### Vectorization
Using NumPy's `np.dot()` instead of explicit loops provides:
- ✅ Significant speed improvements
- ✅ Cleaner, more readable code
- ✅ Better numerical stability

### Gradient Descent
An iterative optimization algorithm that:
1. Computes gradients (direction of steepest descent)
2. Updates parameters by moving in opposite direction
3. Repeats until convergence

### Cost Function Monitoring
Tracking cost across iterations helps:
- Detect convergence
- Identify learning rate issues
- Validate model training

---

## 🔧 Customization

To use your own data or adjust parameters:

```python
# Step 1: Update training data
X_train = np.array([[...], [...], ...])  # Your features
y_train = np.array([...])                # Your targets

# Step 2: Adjust hyperparameters
alpha = 1.0e-6          # Try different learning rates
no_of_iterations = 2000  # Increase for more training

# Step 3: Initialize parameters
initial_w = np.zeros(num_features)
initial_b = 0.0
```

---

## 📚 Learning Outcomes

This project teaches:
- ✅ Multivariate linear regression fundamentals
- ✅ Gradient descent optimization technique
- ✅ Cost function computation and interpretation
- ✅ Vectorized NumPy operations
- ✅ Model training and convergence analysis
- ✅ Parameter optimization strategies

---

## 🎓 Advanced Extensions

Consider implementing:
- Feature scaling/normalization for faster convergence
- Regularization (L1/L2) to prevent overfitting
- Cross-validation for model evaluation
- Testing set prediction
- Scikit-learn comparison
- Mini-batch gradient descent
- Polynomial features

---

## 📝 Notes

- The initial parameters are pre-trained in this example
- The small learning rate ensures stable convergence
- MSE is used as the cost metric
- All computations use double-precision floating point

---

## 🎯 Next Steps

1. Experiment with different learning rates
2. Try different initial parameters
3. Add more training examples
4. Implement feature normalization
5. Create test set predictions
6. Compare with scikit-learn implementation

---

**Language**: Python 3.7+  
**Libraries**: NumPy, Matplotlib  
**Last Updated**: 2025
