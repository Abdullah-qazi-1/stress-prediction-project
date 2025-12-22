import joblib
model = joblib.load('stress_model.pkl')
importances = model.feature_importances_
features = model.feature_names_in_
sorted_idx = importances.argsort()
print('Top 5 features causing stress:')
for i in sorted_idx[-5:][::-1]:
    print(f'{features[i]}: {importances[i]:.4f}')