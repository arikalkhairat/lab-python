import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

file_path = 'dataset.csv'
data = pd.read_csv(file_path, delimiter=';')

numeric_columns = ['Publishing Year', 'Author_Rating', 'Book_average_rating', 'Book_ratings_count',
                   'gross sales', 'publisher revenue', 'sale price', 'sales rank']

data[numeric_columns] = data[numeric_columns].apply(
    pd.to_numeric, errors='coerce')

data = data.dropna(subset=['units sold'])

X = data[numeric_columns]
y = data['units sold']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

columns_without_observed_values = X.columns[X.count() == 0]

numeric_columns = [
    col for col in numeric_columns if col not in columns_without_observed_values]

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train[numeric_columns])
X_test_imputed = imputer.transform(X_test[numeric_columns])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

hist_gradient_boosting_classifier = HistGradientBoostingClassifier()

hist_gradient_boosting_classifier.fit(X_train_scaled, y_train)

y_pred = hist_gradient_boosting_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
