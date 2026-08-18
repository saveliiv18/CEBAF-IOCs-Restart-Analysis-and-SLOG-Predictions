# CEBAF-IOCs-Restart-Analysis-and-SLOG-Predictions
CEBAF IOCs Restart Analysis and SLOG Predictions

A data science and machine learning analysis of CEBAF Input/Output Controller (IOC) restoration events recorded in Jefferson Lab’s electronic logbook (SLOG).

This project was developed by Aston French, Savelii Vorotnikov, and Kinjy Fortunat in August 2026.

Project Overview

The Continuous Electron Beam Accelerator Facility (CEBAF) is controlled by hundreds of Input/Output Controllers (IOCs). When an IOC stops responding, it can be automatically restarted to minimize downtime. Each automatic restart creates an entry in Jefferson Lab’s electronic logbook, SLOG.

This project investigates patterns in these restoration events and evaluates whether temporal information can help distinguish failed restorations from successful restorations.

Research Question

Can temporal features help predict whether an automatic IOC restoration will fail?

This is treated as a binary classification problem:

* Restored (False) — the IOC restoration succeeded.
* Failed (True) — the IOC restoration failed.

The project focuses on both exploratory data analysis (EDA) and machine learning classification.

⸻

Dataset

The analysis uses SLOG records containing information about CEBAF IOC restoration events.

The raw dataset contains fields including:

* lognumber
* title
* body/format
* body/content

The original CSV is included in the repository as:

data/slog_raw_data.csv

The raw data is filtered and processed before analysis, so not every original SLOG record necessarily becomes an observation in the final modeling dataset.

⸻

Data Preparation

The analysis performs several preprocessing and feature-engineering steps:

1. Filters SLOG titles for restoration-related entries.
2. Extracts the target restoration timestamp from the SLOG body/content.
3. Converts extracted timestamps into Python datetime values.
4. Removes records where a valid restoration timestamp cannot be extracted.
5. Creates temporal features from the timestamp.
6. Creates the binary restoration-failure target.

Engineered Features

The primary temporal features include:

* Hour
* Month
* Year
* Season
* Time of day

The target variable is:

restoration_failed

It is set to True when the SLOG title contains “restoration failed” and False for successful restoration entries.

⸻

Exploratory Data Analysis

The analyzed sample contained 1,116 usable restoration events:

* 1,010 successful restorations
* 106 failed restorations

This corresponds to approximately:

* 90.5% successful
* 9.5% failed

Restoration activity was concentrated during daytime hours, particularly around 9:00 AM–12:00 PM.

The dataset also has uneven temporal coverage. Spring and Summer dominate the analyzed sample, while Fall and 2025 contain relatively few observations.

Because restoration failures represent a small minority of the observations, the dataset is class imbalanced. For this reason, accuracy alone is not an adequate measure of model performance.

⸻

Correlation Analysis

A Pearson correlation matrix was used to examine simple linear relationships among:

* Hour
* Month
* Year
* Restoration failure

The correlations between the temporal variables and restoration failure were generally weak.

However, weak Pearson correlation does not establish that temporal information is irrelevant. Variables such as hour and month are cyclical, and their relationships with restoration failure may be nonlinear.

This motivated comparing both a linear and nonlinear classifier.

⸻

Machine Learning Approach

Two classification models were evaluated:

Logistic Regression

Logistic Regression provides an interpretable baseline for modeling simpler relationships between temporal features and restoration outcomes.

Random Forest

A Random Forest Classifier was used to capture nonlinear relationships and interactions that may not be represented effectively by a linear model.

Train/Test Split

The dataset was divided into:

* 80% training data
* 20% testing data

A stratified split was used to preserve the proportion of successful and failed restorations across the training and testing sets.

The models were evaluated on the same test observations to allow a fair comparison.

Because restoration failures are substantially less common than successful restorations, class weighting was used to give the minority failure class greater influence during training.

⸻

Evaluation Metrics

Model performance was evaluated using several metrics rather than accuracy alone.

Metric	Meaning
Accuracy	Percentage of all predictions that were correct
Precision	Of the events predicted as failures, how many actually failed
Recall	Of the actual failures, how many the model detected
F1 Score	Balance between precision and recall
PR-AUC	Measures precision-recall performance across classification thresholds
Confusion Matrix	Shows correct and incorrect predictions for each class

Recall is particularly important when evaluating the ability to identify the relatively uncommon restoration failures.

⸻

Model Results

Logistic Regression

Metric	Result
Accuracy	83.48%
Precision	16.67%
Recall	19.10%
F1 Score	17.78%
PR-AUC	27.10%

Confusion matrix:

	Predicted Restored	Predicted Failed
Actually Restored	183	20
Actually Failed	17	4

The Logistic Regression model correctly identified only 4 of the 21 failures in the test set. Although it achieved relatively high overall accuracy, its low failure recall demonstrates why accuracy can be misleading for an imbalanced dataset.

Random Forest

Metric	Result
Accuracy	63.39%
Precision	16.48%
Recall	71.43%
F1 Score	26.79%
PR-AUC	16.78%

Confusion matrix:

	Predicted Restored	Predicted Failed
Actually Restored	127	76
Actually Failed	6	15

The Random Forest detected 15 of the 21 failures, corresponding to 71.43% recall, but produced substantially more false-positive failure predictions.

⸻

Model Comparison

The two models demonstrate an important tradeoff.

Logistic Regression achieved higher overall accuracy and produced fewer false alarms, but it failed to detect most actual restoration failures.

Random Forest detected substantially more failures, but this improvement in recall came at the cost of many more false positives and lower overall accuracy.

Neither model provides sufficiently strong evidence for reliable operational failure prediction using temporal features alone.

⸻

Key Findings

The analysis suggests that:

* IOC restoration activity shows temporal patterns.
* Restoration activity is concentrated during certain daytime hours.
* Failure rates vary across temporal categories.
* Temporal features contain some predictive signal.
* Logistic Regression struggles to detect the minority failure class.
* Random Forest substantially improves failure recall but generates many false alarms.
* Temporal variables alone are insufficient for reliable IOC restoration-failure prediction.

⸻

Limitations

Several limitations are important when interpreting these results.

Class Imbalance

Only 106 of 1,116 analyzed restoration events were failures. This makes failure detection substantially more difficult and means that accuracy alone can provide a misleading assessment of model quality.

Uneven Temporal Coverage

Most observations are concentrated in Spring and Summer 2026.

Fall and 2025 contain very small samples, so failure rates calculated from those groups should be interpreted cautiously.

Limited Predictive Features

The current models primarily describe when an event occurred.

They do not incorporate potentially important operational information such as:

* IOC identity
* subsystem
* system load
* previous failure history
* other system-state information

Correlation Limitations

Pearson correlation measures linear relationships. Temporal variables such as hour and month are cyclical, and potentially useful nonlinear relationships may therefore not appear as strong Pearson correlations.

⸻

Future Work

Future analysis could investigate whether additional verified operational features improve failure prediction.

Potential areas for investigation include:

* IOC-specific behavior
* subsystem information
* system-state variables
* IOC restart history
* prior failure patterns
* additional operational features

A larger dataset with broader and more balanced temporal coverage would also provide stronger evidence for evaluating long-term patterns.

⸻

Repository Structure

CEBAF-IOCs-Restart-Analysis-and-SLOG-Predictions/
│
├── README.md
├── data/
│   └── slog_raw_data.csv
├── notebooks/
│   └── CEBAF_IOCs_Restart_Analysis_and_SLOG_Predictions.ipynb
├── src/
│   └── analysis.py
└── outputs/
    └── generated figures and results

The exact structure may vary depending on whether the analysis is executed primarily through the original notebook or the standalone Python script.

⸻

Running the Analysis

The original analysis is available through the project’s Google Colab notebook:

https://colab.research.google.com/drive/1besEKXja4Qvo1EgzFOp0m6XJaN-2kmj8#scrollTo=hEcPCIMpA_ze

The repository version of the code is intended to preserve the analysis performed in the original notebook.

When running locally, ensure that the required Python data-science dependencies are installed and that:

data/slog_raw_data.csv

is available at the expected path.

⸻

Authors

Aston French
Savelii Vorotnikov
Kinjy Fortunat

⸻

Acknowledgements

This project was developed in connection with Brookhaven National Laboratory and Jefferson Lab.

The project was supported by the U.S. Department of Energy, Office of Science, Workforce Development for Teachers and Scientists (WDTS).

The analysis and conclusions presented in this repository should not be interpreted as an official operational model or official endorsement by Brookhaven National Laboratory, Jefferson Lab, or the U.S. Department of Energy.
