import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("STUDENT PERFORMANCE ANALYSIS")
print("=" * 60)

# Load the dataset
df = pd.read_csv('student_performance.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")

# Basic info about the dataset
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Target variable analysis (focusing on G3 - final grade)
print("\nTarget Variable (G3 - Final Grade) Analysis:")
print(f"Mean: {df['G3'].mean():.2f}")
print(f"Std: {df['G3'].std():.2f}")
print(f"Min: {df['G3'].min()}")
print(f"Max: {df['G3'].max()}")

# Correlation analysis
print("\nCorrelation with Final Grade (G3):")
correlations = df.select_dtypes(include=[np.number]).corr()['G3'].sort_values(ascending=False)
print(correlations.head(10))

# Data preprocessing
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Separate features and target
X = df.drop(['G1', 'G2', 'G3'], axis=1)  # Remove all grade columns for pure prediction
y = df['G3']  # Final grade as target

# Encode categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"\nCategorical columns to encode: {list(categorical_columns)}")

label_encoders = {}
X_encoded = X.copy()

for col in categorical_columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Feature selection
print(f"\nOriginal features: {X_encoded.shape[1]}")
selector = SelectKBest(score_func=f_regression, k=15)  # Select top 15 features
X_selected = selector.fit_transform(X_encoded, y)
selected_features = X_encoded.columns[selector.get_support()].tolist()
print(f"Selected features ({len(selected_features)}): {selected_features}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=3)
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Model training and evaluation
print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 30)
    
    # Use scaled data for Linear Regression, original for tree-based models
    if name == 'Linear Regression':
        X_train_model, X_test_model = X_train_scaled, X_test_scaled
    else:
        X_train_model, X_test_model = X_train, X_test
    
    # Train the model
    model.fit(X_train_model, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_model)
    y_pred_test = model.predict(X_test_model)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_test
    }
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance for Random Forest
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': [selected_features[i] for i in range(len(selected_features))],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Model comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_df = pd.DataFrame(results).T
print(comparison_df[['test_r2', 'test_rmse', 'test_mae', 'cv_mean']])

# Find best model
best_model_name = comparison_df['test_r2'].idxmax()
best_score = comparison_df.loc[best_model_name, 'test_r2']

print(f"\nBest Model: {best_model_name}")
print(f"Best Test R² Score: {best_score:.4f}")

# Create visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')

# 1. Target distribution
axes[0, 0].hist(y, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Final Grades (G3)')
axes[0, 0].set_xlabel('Final Grade')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
axes[0, 0].legend()

# 2. Model performance comparison
model_names = list(results.keys())
test_r2_scores = [results[name]['test_r2'] for name in model_names]
colors = ['lightcoral', 'lightgreen', 'lightskyblue']

bars = axes[0, 1].bar(model_names, test_r2_scores, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Model Performance Comparison (Test R²)')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_ylim(0, 1)

# Add value labels on bars
for bar, score in zip(bars, test_r2_scores):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Actual vs Predicted (Best Model)
best_predictions = results[best_model_name]['predictions']
axes[1, 0].scatter(y_test, best_predictions, alpha=0.6, color='green')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})')
axes[1, 0].set_xlabel('Actual Final Grade')
axes[1, 0].set_ylabel('Predicted Final Grade')
axes[1, 0].text(0.05, 0.95, f'R² = {best_score:.3f}', transform=axes[1, 0].transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. Feature importance (top 10)
top_features = feature_importance.head(10)
axes[1, 1].barh(range(len(top_features)), top_features['importance'], color='orange', alpha=0.7)
axes[1, 1].set_yticks(range(len(top_features)))
axes[1, 1].set_yticklabels(top_features['feature'])
axes[1, 1].set_title('Top 10 Feature Importance (Random Forest)')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('student_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'student_performance_analysis.png'")

# Save results to CSV
results_summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Test_R2': [results[name]['test_r2'] for name in results.keys()],
    'Test_RMSE': [results[name]['test_rmse'] for name in results.keys()],
    'Test_MAE': [results[name]['test_mae'] for name in results.keys()],
    'CV_Mean_R2': [results[name]['cv_mean'] for name in results.keys()]
})

results_summary.to_csv('model_results.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("Results saved to 'model_results.csv'")
print("Feature importance saved to 'feature_importance.csv'")

# Key insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print(f"1. Dataset contains {df.shape[0]} students with {df.shape[1]} attributes")
print(f"2. Final grades range from {y.min()} to {y.max()} with mean {y.mean():.2f}")
print(f"3. Best performing model: {best_model_name} with R² = {best_score:.4f}")
print(f"4. Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"5. Top 3 predictive features: {', '.join(feature_importance.head(3)['feature'].tolist())}")

# Grade categories analysis
print(f"6. Grade distribution:")
grade_categories = pd.cut(y, bins=[0, 10, 14, 20], labels=['Poor (0-10)', 'Average (11-14)', 'Good (15-20)'])
print(grade_categories.value_counts())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("Files generated:")
print("- student_performance_analysis.png (visualizations)")
print("- model_results.csv (model performance)")
print("- feature_importance.csv (feature rankings)")
print("=" * 60)
