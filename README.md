# Pattern Recognition Assignment - Iris Dataset Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📚 Project Overview

This repository contains **Assignment-1** for Pattern Recognition course, implementing a **k-Nearest Neighbors (k-NN)** algorithm for classifying iris flower species using the famous UCI Iris dataset. The project demonstrates a complete machine learning pipeline from data loading to model evaluation and visualization.

### 🎯 Objectives
- Implement k-NN classification algorithm for multi-class pattern recognition
- Perform comprehensive exploratory data analysis (EDA)
- Optimize hyperparameters using cross-validation
- Evaluate model performance with multiple metrics
- Generate professional visualizations for data insights

### 📊 Dataset Information
- **Source**: UCI Machine Learning Repository
- **Samples**: 150 iris flowers (50 per species)
- **Features**: 4 morphological measurements
  - Sepal Length (cm)
  - Sepal Width (cm) 
  - Petal Length (cm)
  - Petal Width (cm)
- **Classes**: 3 species
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica

## 🏆 Results Summary

| Metric | Value |
|--------|-------|
| **Model Accuracy** | **95.56%** |
| **Optimal k-value** | 9 |
| **Training Samples** | 105 |
| **Testing Samples** | 45 |
| **Most Important Feature** | Petal Width (0.260) |

### 📈 Classification Report
```
                 precision    recall  f1-score   support
    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       0.88      1.00      0.94        15
 Iris-virginica       1.00      0.87      0.93        15

       accuracy                           0.96        45
```

## 📁 Repository Structure

```
iris-classification/
├── 📄 assignment_1.py          # Main Python script
├── 📓 assignment_1.ipynb       # Jupyter notebook version
├── 📋 requirements.txt         # Python dependencies
├── 📖 README.md               # This file
├── 📄 project_report.pdf      # Detailed assignment report
├── 🖼️ visualizations/         # Generated plots and charts
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── pair_plot.png
│   ├── confusion_matrix.png
│   ├── decision_boundary.png
│   ├── feature_importance.png
│   └── optimal_k_selection.png
└── 💾 models/                 # Saved model artifacts
    ├── knn_iris_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/iris-classification.git
   cd iris-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the assignment**
   ```bash
   python assignment_1.py
   ```

### 📓 Jupyter Notebook Usage
```bash
jupyter notebook assignment_1.ipynb
```

## 🔬 Technical Implementation

### Algorithm Features
- **k-NN Classification**: Non-parametric, instance-based learning
- **Cross-Validation**: 10-fold CV for optimal k-value selection
- **Feature Scaling**: StandardScaler for distance-based calculations
- **Stratified Sampling**: Maintains class distribution in train/test split

### Key Components

#### 1. Data Preprocessing
```python
# Feature scaling for k-NN algorithm
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label encoding for target classes
le = LabelEncoder()
y_encoded = le.fit_transform(y['class'])
```

#### 2. Hyperparameter Optimization
```python
# Cross-validation for optimal k selection
k_values = list(range(1, 31))
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10)
    cv_scores.append(scores.mean())
```

#### 3. Model Evaluation
- **Accuracy Score**: 95.56%
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: Permutation-based analysis
- **Cross-Validation**: Robust performance estimation

## 📊 Visualizations Generated

1. **Class Distribution**: Balanced dataset verification
2. **Correlation Heatmap**: Feature relationship analysis
3. **Pair Plot**: Multi-dimensional data exploration
4. **Confusion Matrix**: Classification performance matrix
5. **Decision Boundary**: 2D visualization of classification regions
6. **Feature Importance**: Ranking of predictive features
7. **Optimal k Selection**: Cross-validation results

## 🎓 Educational Outcomes

### Machine Learning Concepts Demonstrated
- ✅ **Supervised Learning**: Classification with labeled data
- ✅ **Distance-Based Learning**: k-NN algorithm implementation
- ✅ **Cross-Validation**: Model selection and validation
- ✅ **Feature Engineering**: Scaling and preprocessing
- ✅ **Performance Evaluation**: Multiple metrics analysis
- ✅ **Data Visualization**: Comprehensive EDA techniques

### Statistical Analysis
- Descriptive statistics and data quality assessment
- Correlation analysis between features
- Feature importance ranking using permutation testing
- Residual analysis for model validation

## 🛠️ Dependencies

### Core Libraries
- **NumPy** (≥1.21.0): Numerical computing
- **Pandas** (≥1.3.0): Data manipulation and analysis
- **scikit-learn** (≥1.0.0): Machine learning algorithms
- **ucimlrepo** (≥0.0.7): UCI dataset repository access

### Visualization
- **Matplotlib** (≥3.4.0): Basic plotting functionality
- **Seaborn** (≥0.11.0): Statistical data visualization

### Utilities
- **joblib** (≥1.1.0): Model serialization and persistence

## 📈 Performance Metrics

| Species | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Setosa | 1.00 | 1.00 | 1.00 | 15 |
| Versicolor | 0.88 | 1.00 | 0.94 | 15 |
| Virginica | 1.00 | 0.87 | 0.93 | 15 |
| **Overall** | **0.96** | **0.96** | **0.96** | **45** |

## 🔍 Key Findings

1. **Petal measurements** are more discriminative than sepal measurements
2. **Length of membership** shows strongest correlation with target classes
3. **k=9** provides optimal balance between bias and variance
4. **Perfect classification** achieved for Iris Setosa species
5. **Minor confusion** exists between Versicolor and Virginica species

## 🚀 Future Enhancements

- [ ] Implement other classification algorithms (SVM, Random Forest)
- [ ] Add hyperparameter tuning for distance metrics
- [ ] Include ensemble methods for improved accuracy
- [ ] Develop web interface for interactive classification
- [ ] Add support for real-time prediction API

## 📝 Assignment Requirements

✅ **Data Loading**: UCI Iris dataset integration  
✅ **EDA**: Comprehensive statistical and visual analysis  
✅ **Preprocessing**: Feature scaling and encoding  
✅ **Algorithm**: k-NN implementation with optimization  
✅ **Evaluation**: Multiple performance metrics  
✅ **Visualization**: Professional plots and charts  
✅ **Documentation**: Complete code documentation  
✅ **Report**: Detailed technical report (PDF)  

## 👨‍🎓 Author Information

**Course**: Pattern Recognition  
**Institution**: Indian Institute of Technology Patna  
**Assignment**: Assignment-1 - Iris Dataset Classification  
**Algorithm**: k-Nearest Neighbors (k-NN)  
**Date**: September 2025  

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📞 Contact

If you have any questions about this assignment implementation, please feel free to reach out or create an issue in this repository.

---

### ⭐ If you found this assignment helpful, please consider giving it a star!

**Happy Learning! 🎯📚**
