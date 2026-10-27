from pathlib import Path
import joblib
import pandas as pd
MODEL_PATH=Path(__file__).resolve().parents[1]/'models'/'trained_model.joblib'
def load_model(path=MODEL_PATH):
    if not Path(path).exists(): raise FileNotFoundError(f'Model file not found: {path}')
    return joblib.load(path)
def make_prediction_input(values,bundle):
    features=bundle['features']; row={k:values.get(k) for k in features}; frame=pd.DataFrame([row],columns=features)
    for col in bundle.get('numerical_features',[]):
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
        if frame[col].isna().any(): raise ValueError(f'{col} must be a valid number.')
        if (frame[col]<0).any(): raise ValueError(f'{col} cannot be negative.')
    return frame

def predict_one(values, bundle):
    X = make_prediction_input(values, bundle)
    pipe = bundle['model']
    cls = int(pipe.predict(X)[0])
    prob = None

    if hasattr(pipe, 'predict_proba'):
        classes = list(pipe.classes_)
        if 1 in classes:
            prob = float(pipe.predict_proba(X)[0][classes.index(1)])

    return {
        'prediction': cls,
        'probability': prob,
        'input': X
    }
