import pandas as pd, joblib, numpy as np, sys

def predict_new(csv_path, model_path='fba_output/final_model.joblib', threshold_path='fba_output/threshold.joblib'):
    pipe = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    df_new = pd.read_csv(csv_path)
    # Ensure new data has same feature columns
    expected_features = pipe.named_steps['prep'].get_feature_names_out()
    Xt = pipe.named_steps['prep'].transform(df_new)
    probs = pipe.named_steps['model'].predict_proba(Xt)[:,1]
    preds = (probs >= threshold).astype(int)
    df_new['Risk_Probability'] = probs
    df_new['Predicted_High_Risk'] = preds
    out_path = 'predicted_output.csv'
    df_new.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    csv_in = sys.argv[1] if len(sys.argv) > 1 else 'new_data.csv'
    predict_new(csv_in)
