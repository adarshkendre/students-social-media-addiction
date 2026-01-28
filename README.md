# 📱 Social Media Usage & Mental Health Analytics Dashboard
Live Link : https://students-social-media-addiction.streamlit.app/

> An advanced interactive dashboard for analyzing student social media usage patterns, mental health impacts, and academic performance using data science and machine learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🌟 Features

### 📊 **8 Interactive Analysis Tabs**

1. **Overview Dashboard**
   - Real-time metrics with animated gauges
   - Distribution analysis with violin plots
   - Platform popularity sunburst charts
   - Simulated usage pattern trends

2. **Demographics Analysis**
   - Age pyramid visualizations
   - Geographic treemaps
   - Academic level breakdowns
   - 3D demographic explorer

3. **Usage Patterns**
   - Multi-dimensional pattern analysis
   - Sleep vs. usage correlations
   - Platform comparison charts
   - Parallel coordinates visualization

4. **Mental Health Analysis**
   - Risk zone identification
   - Platform impact studies
   - Conflict analysis
   - Interactive radar charts

5. **Academic Impact**
   - Performance correlation analysis
   - Sankey flow diagrams
   - Comparative studies
   - Multi-factor impact assessment

6. **Correlation & Statistics**
   - Interactive correlation heatmaps
   - Scatter plot matrices
   - Statistical significance tests (T-test, ANOVA, Pearson)
   - Regression analysis with R² values

7. **Machine Learning Insights**
   - Principal Component Analysis (PCA)
   - K-Means clustering with auto-profiling
   - Feature importance analysis
   - Intelligent pattern recognition

8. **Geographic View**
   - World map visualizations
   - Cross-country comparisons
   - Bubble charts
   - Parallel categories flow

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/social-media-analytics.git
cd social-media-analytics
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas matplotlib seaborn plotly numpy scipy scikit-learn
```

3. **Prepare your data**
   - Place your CSV file named `Students Social Media Addiction.csv` in the project directory
   - Or use the built-in sample data generator (no file needed!)

4. **Run the dashboard**
```bash
streamlit run dashboard.py
```

5. **Open in browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If not, copy the URL from the terminal

---

## 📁 Dataset Requirements

### Required Columns

Your CSV file should contain the following columns:

| Column Name | Type | Description |
|------------|------|-------------|
| `Student_ID` | String | Unique identifier for each student |
| `Age` | Integer | Student's age (16-30) |
| `Gender` | String | Male/Female/Other |
| `Academic_Level` | String | High School/Undergraduate/Graduate/PhD |
| `Country` | String | Student's country |
| `Avg_Daily_Usage_Hours` | Float | Daily social media usage in hours |
| `Most_Used_Platform` | String | Primary social media platform |
| `Affects_Academic_Performance` | String | Yes/No |
| `Sleep_Hours_Per_Night` | Float | Average sleep duration |
| `Mental_Health_Score` | Integer | Score from 0-100 (higher is better) |
| `Relationship_Status` | String | Single/In a relationship/Married |
| `Conflicts_Over_Social_Media` | String | Yes/No |
| `Addicted_Score` | Integer | Addiction level 0-100 (higher is worse) |

### Sample Data Format

```csv
Student_ID,Age,Gender,Academic_Level,Country,Avg_Daily_Usage_Hours,Most_Used_Platform,Affects_Academic_Performance,Sleep_Hours_Per_Night,Mental_Health_Score,Relationship_Status,Conflicts_Over_Social_Media,Addicted_Score
STD0001,20,Male,Undergraduate,USA,5.5,Instagram,Yes,6.5,65,Single,No,55
STD0002,22,Female,Graduate,India,7.2,TikTok,Yes,5.8,58,In a relationship,Yes,72
```

---

## 🎯 Key Features Explained

### 🔍 Dynamic Filtering

Filter data in real-time using:
- Gender selection
- Academic level (multi-select)
- Platform preferences
- Age range slider
- Usage hours range

### 📈 Advanced Visualizations

- **3D Scatter Plots** - Explore multi-dimensional relationships
- **Violin Plots** - See distribution patterns with statistical details
- **Sunburst Charts** - Hierarchical data visualization
- **Heatmaps** - Correlation and intensity matrices
- **Sankey Diagrams** - Flow and relationship mapping
- **Radar Charts** - Multi-metric comparisons
- **Animated Gauges** - Real-time metric displays

### 🤖 Machine Learning

**PCA (Principal Component Analysis)**
- Reduces data to 2 main components
- Shows variance explained by each component
- Reveals hidden patterns in behavior

**K-Means Clustering**
- Groups students with similar behaviors
- Auto-generates cluster profiles
- Identifies risk categories:
  - 🔴 High Risk
  - 🟡 Heavy Users
  - 🔵 Moderate Users
  - 🟢 Healthy Users

### 📊 Statistical Tests

- **T-Tests** - Gender differences in addiction
- **ANOVA** - Platform differences in mental health
- **Pearson Correlation** - Relationship strength measurement
- **Regression Analysis** - Predictive modeling with R² values

### 💡 Smart Insights

The dashboard automatically generates:
- Usage alerts and warnings
- Mental health risk assessments
- Academic impact summaries
- Personalized recommendations
- Correlation interpretations

---

## 📥 Export Options

Export your filtered data in multiple formats:

1. **CSV Export** - Filtered dataset with all records
2. **Summary Statistics** - Descriptive statistics report
3. **Correlation Matrix** - Complete correlation analysis

---

## 🎨 Personalized Analysis

Get personalized insights by entering your own data:

1. Navigate to "Personalized Insights" section
2. Input your usage hours, sleep, mental health score
3. Click "Analyze My Profile"
4. Receive customized recommendations
5. Compare with similar students

---

## 🛠️ Advanced Options

Unlock power user features:

- **Show Raw Statistics** - View dataset metadata
- **Clear Cache** - Reset cached computations
- **Performance Metrics** - Monitor app performance
- **Custom Clustering** - Adjust number of clusters
- **Filter Analytics** - Track filter impact

---

## 📚 Use Cases

### For Researchers
- Analyze social media impact on mental health
- Study correlation patterns
- Generate publication-ready visualizations
- Export statistical test results

### For Educators
- Monitor student wellbeing trends
- Identify at-risk students
- Support digital wellness programs
- Create awareness presentations

### For Students
- Understand your own usage patterns
- Get personalized recommendations
- Compare with peers
- Track digital wellbeing goals

### For Parents
- Learn about teen social media habits
- Understand risk factors
- Get family-oriented recommendations
- Monitor healthy usage benchmarks

---

## 🧪 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Interactive web application framework |
| **Pandas** | Data manipulation and analysis |
| **Plotly** | Interactive visualizations |
| **Seaborn/Matplotlib** | Statistical plotting |
| **Scikit-learn** | Machine learning (PCA, K-Means) |
| **NumPy** | Numerical computations |
| **SciPy** | Statistical tests |

---

## 📖 Documentation

### Running with Custom Data

```python
# Option 1: Place your CSV in the same directory
# Name it: Students Social Media Addiction.csv

# Option 2: Use sample data
# Check "Use Sample Data" in the sidebar
```

### Customizing the Dashboard

```python
# Modify number of clusters
n_clusters = st.slider("Number of clusters", 2, 6, 3)

# Change color schemes
color_continuous_scale='Viridis'  # Options: Viridis, RdYlGn, Blues, etc.

# Adjust figure heights
fig.update_layout(height=600)  # Change as needed
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Dashboard won't start
```bash
# Solution: Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Issue:** Data file not found
```bash
# Solution: Use sample data or check file path
# Enable "Use Sample Data" in sidebar
```

**Issue:** Visualizations not rendering
```bash
# Solution: Clear browser cache or try different browser
# Press Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
```

**Issue:** Slow performance
```bash
# Solution: Filter data to reduce records
# Use the filter options in the sidebar
```
