import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df['INSPECTION DATE'] = pd.to_datetime(df['INSPECTION DATE'], errors='coerce')
    df = df.sort_values(['CAMIS', 'INSPECTION DATE'])
    df['SCORE'] = pd.to_numeric(df['SCORE'], errors='coerce')
    df['Y'] = (df['SCORE'] >= 14).astype(int)
    feats = []
    for camis, sub in df.groupby('CAMIS'):
        sub = sub.sort_values('INSPECTION DATE')
        sub['n_past_inspections'] = range(len(sub))
        sub['days_since_last_inspection'] = sub['INSPECTION DATE'].diff().dt.days.fillna(9999)
        sub['avg_score_last_3'] = sub['SCORE'].rolling(3, min_periods=1).mean().shift(1)
        sub['max_score_last_5'] = sub['SCORE'].rolling(5, min_periods=1).max().shift(1)
        sub['count_past_high_score'] = (sub['SCORE'].shift(1) >= 14).expanding().sum().fillna(0)
        sub['score_diff_last2'] = sub['SCORE'].diff().fillna(0)
        sub['is_reinspection'] = sub['INSPECTION TYPE'].str.contains('Re-', case=False, na=False).astype(int)
        feats.append(sub)
    df = pd.concat(feats, ignore_index=True)
    df['cuisine_group'] = df['CUISINE DESCRIPTION'].fillna('Unknown')
    df['boro'] = df['BORO'].fillna('Unknown')
    df['is_chain'] = df.groupby('DBA')['CAMIS'].transform('nunique') > 3
    num_cols = ['n_past_inspections','days_since_last_inspection','avg_score_last_3','max_score_last_5',
                'count_past_high_score','score_diff_last2']
    cat_cols = ['cuisine_group','boro','is_reinspection','is_chain']
    df = df.dropna(subset=['Y'])
    return df, num_cols, cat_cols

def train_model(csv_path, out_dir):
    df, num_cols, cat_cols = load_and_engineer(csv_path)
    cutoff_test = df['INSPECTION DATE'].quantile(0.8)
    df_train = df[df['INSPECTION DATE'] < cutoff_test]
    df_test = df[df['INSPECTION DATE'] >= cutoff_test]
    X_train, y_train = df_train[num_cols+cat_cols], df_train['Y']
    X_test, y_test = df_test[num_cols+cat_cols], df_test['Y']
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),('enc', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    print(classification_report(y_test, preds))
    auc = roc_auc_score(y_test, probs)
    print("AUC:", auc)
    prec, rec, thr = precision_recall_curve(y_test, probs)
    target_recall = 0.9
    idxs = np.where(rec >= target_recall)[0]
    threshold = 0.5
    if len(idxs) > 0:
        best_idx = idxs[np.argmax(prec[idxs])]
        threshold = thr[best_idx-1] if best_idx>0 else 1.0
        print(f"Threshold achieving recall >= {target_recall}: {threshold:.3f} (precision {prec[best_idx]:.3f}, recall {rec[best_idx]:.3f})")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, 'final_model.joblib'))
    joblib.dump(threshold, os.path.join(out_dir, 'threshold.joblib'))
    print("Model and threshold saved to", out_dir)

if __name__ == "__main__":
    train_model('FBA Coursework Data.csv', 'fba_output')
