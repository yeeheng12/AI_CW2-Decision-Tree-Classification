# COMP2611 Artificial Intelligence Coursework 2 – Decision Tree Classification

## Overview

This project implements a **Decision Tree Classification** model using Python and Scikit-learn as part of the COMP2611 Artificial Intelligence coursework at the University of Leeds.

The project focuses on applying machine learning techniques to a classification problem, covering the complete workflow from data loading and preprocessing to model training, evaluation and optimisation.

It also explores decision tree pruning and feature selection to understand how model complexity affects classification performance.

## Objectives

The main objectives of this project are to:

- Load and process structured datasets using Pandas and NumPy.
- Identify and remove records containing missing or invalid values.
- Perform statistical analysis on dataset features.
- Split data into training and testing sets.
- Train and evaluate Decision Tree Classification models.
- Apply cost-complexity pruning to control model complexity.
- Identify important features and investigate their contribution to classification.
- Analyse decision tree depth and model performance.

## Key Features

### 1. Data Loading and Preprocessing
- Load CSV datasets using Pandas.
- Convert data into NumPy arrays for numerical processing.
- Extract column headers and dataset dimensions.
- Remove records containing invalid values (`-99`).

### 2. Statistical Analysis
- Calculate the mean and standard deviation of numerical features.
- Calculate the coefficient of variation for each feature.
- Examine the distribution and variability of dataset features.

### 3. Model Training
- Implement a Decision Tree Classifier using Scikit-learn.
- Split the dataset into training and testing subsets.
- Apply stratified sampling to maintain class proportions.
- Train models using different cost-complexity pruning parameters (`ccp_alpha`).

### 4. Model Evaluation
Evaluate classification performance using:
- Accuracy
- Recall

Generate predictions on unseen test data and compare the performance of unpruned and pruned decision trees.

### 5. Decision Tree Optimisation
- Determine an appropriate cost-complexity pruning parameter.
- Compare tree structures before and after pruning.
- Measure decision tree depth.
- Identify the most important feature used by the decision tree.
- Investigate pruning using the most important feature individually.
- Explore optimal tree depth using the two most important features.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computation and data preprocessing |
| Scikit-learn | Decision tree modelling, data splitting and evaluation |

## Project Structure

```text
AI_CW2/
│
├── CW2 (1).py
│   └── Main Python implementation
│
├── Assessment Brief (1).pdf
│   └── Coursework requirements
│
└── DT.csv
    └── Input dataset (not included in repository)
```

## Getting Started

### Prerequisites

Ensure Python 3.8 or later is installed.

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn
```

### Dataset Setup

The program expects a CSV dataset named `DT.csv`.

Place the dataset in the project directory, or update the `file_path` variable in the main Python script to point to your dataset location.

```python
file_path = "DT.csv"
```

**Note:** The dataset is not included in this repository.

### Running the Project

Clone the repository:

```bash
git clone https://github.com/yeeheng12/AI_CW2.git
```

Navigate into the project directory:

```bash
cd AI_CW2
```

Install the dependencies:

```bash
pip install pandas numpy scikit-learn
```

Run the Python script:

```bash
python "CW2 (1).py"
```

## Learning Outcomes

Through this coursework, I gained practical experience in:

- Understanding and implementing supervised machine learning algorithms.
- Working with structured datasets and performing data preprocessing.
- Applying statistical techniques to analyse numerical features.
- Training and evaluating classification models.
- Understanding overfitting and the importance of model complexity.
- Applying cost-complexity pruning to improve decision tree generalisation.
- Investigating feature importance and interpreting decision tree structures.

## Author

**Loke Yee Heng**

---

*University of Leeds – COMP2611 Artificial Intelligence Coursework 2.*
