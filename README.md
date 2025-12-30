# Cost-Sensitive-Credit-Risk-Modeling-and-Business-Metric-Alignment
This project addresses the problem of Credit Card Default prediction using a Business-Value-First approach.
Rather than optimizing for standard metrics like Accuracy or F1-Score, this pipeline focuses on minimizing the specific financial
costs associated with False Negatives (failing to identify a defaulter).
The analysis is structured into four critical stages to ensure the model is both predictive and financially sound:

1- Automated Hyperparameter Tuning: Using Optuna to explore the search space for LightGBM and Logistic Regression.

2- Probability Calibration: Using CalibratedClassifierCV (Isotonic/Sigmoid) to ensure that the predicted probabilities reflect real-world default frequencies.

3- Threshold Optimization: Identifying the optimal decision cutoff by minimizing a custom cost function where a False Negative is penalized 3x more than a False Positive.

4- Robust Evaluation: Visualizing the Calibration Curve and Business Cost Curve to validate the model's reliability.
