import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib, os

DATA_PATH = 'data/feat.parquet'
MODEL_PATH = 'models/lgb_trend.txt'
SKLEARN_DUMP = 'models/lgb_trend.pkl'

def main():
    df = pd.read_parquet(DATA_PATH)
    features = [c for c in df.columns if c not in ['y']]
    X, y = df[features], df['y']

    tscv = TimeSeriesSplit(n_splits=5)
    best_model, best_auc = None, -1.0
    for fold, (tr, va) in enumerate(tscv.split(X), start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        mdl = lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.8)
        mdl.fit(Xtr, ytr)
        proba = mdl.predict_proba(Xva)[:,1]
        auc = roc_auc_score(yva, proba)
        print(f'Fold {fold} AUC: {auc:.4f}')
        if auc > best_auc:
            best_auc, best_model = auc, mdl

    os.makedirs('models', exist_ok=True)
    best_model.booster_.save_model(MODEL_PATH)
    joblib.dump(best_model, SKLEARN_DUMP)
    print(f'Best AUC: {best_auc:.4f}. Saved {MODEL_PATH} and {SKLEARN_DUMP}.')

if __name__ == '__main__':
    main()
