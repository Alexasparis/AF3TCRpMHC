#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import argparse
import joblib

def train_rf(train_df, test_df, seed=42, thresholds=None):
    """
    Train a Random Forest multiclass classifier on 4 quality labels: HQ, MQ, AQ, LQ.
    Optionally apply probability thresholds to change predicted labels.

    Parameters:
    - train_df: DataFrame with training features and labels
    - test_df: DataFrame with test features and labels
    - seed: random seed
    - thresholds: dict of per-class probability thresholds, e.g. {'LQ':0.4,'AQ':0.7,'MQ':0.7,'HQ':0.8}

    Returns:
    - model: trained RandomForestClassifier
    - predictions: DataFrame with test set predictions, probabilities, and thresholded labels
    """

    features = [
        'global_plddt', 'plddt_cdr1a', 'plddt_cdr2a', 'plddt_cdr3a',
        'plddt_cdr1b', 'plddt_cdr2b', 'plddt_cdr3b',
        'iptm_mean', 'iptm_tcrpmhc', 'pdockq_AB',
        'avgipde', 'avgipae', 'avgpdockq2', 'iPSAE'
    ]

    label_col = 'Quality'

    # Split features and labels
    X_train = train_df[features]
    y_train = train_df[label_col]
    X_test = test_df[features]

    # Initialize Random Forest
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced',
        n_estimators=1000,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='log2'
    )

    # Train model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    predictions = test_df.copy()
    predictions["predicted_label"] = y_pred

    proba_df = pd.DataFrame(
        y_proba,
        columns=[f"proba_{cls}" for cls in model.classes_],
        index=test_df.index
    )
    predictions = pd.concat([predictions, proba_df], axis=1)

    if thresholds is not None:
        classes = ['LQ','AQ','MQ','HQ']
        for cls in classes:
            if cls not in thresholds:
                raise ValueError(f"Threshold missing for class {cls}")
            if cls == 'LQ':
                predictions[f'pass_{cls}'] = (predictions[f'proba_{cls}'] >= thresholds[cls]).astype(int)
            elif cls == 'AQ':
                predictions[f'pass_{cls}'] = (predictions[f'proba_{cls}'] >= thresholds[cls]).astype(int)
            elif cls == 'MQ':
                predictions[f'pass_{cls}'] = (predictions[f'proba_{cls}'] >= thresholds[cls]).astype(int)
            else:  # HQ
                predictions[f'pass_{cls}'] = (predictions[f'proba_{cls}'] >= thresholds[cls]).astype(int)

        def choose_label(row):
            passing = [cls for cls in classes if row[f'pass_{cls}']==1]
            if len(passing) == 0:
                return 'LQ'  
            else:
                probs = {cls: row[f'proba_{cls}'] for cls in passing}
                return max(probs, key=probs.get)

        predictions['predicted_label_thresholded'] = predictions.apply(choose_label, axis=1)

        predictions['pass_filter'] = predictions['predicted_label_thresholded'].isin(['AQ','MQ','HQ']).astype(int)

    return model, predictions

def prepare_features(df):
    """
    Normaliza las columnas para el RF. Si no existen, las crea desde equivalentes.
    """
    df_new = df.copy()
    
    # Columnas directas
    mapping = {
        'global_plddt': 'plddt',
        'plddt_cdr1a': 'cdr1a_plddt',
        'plddt_cdr2a': 'cdr2a_plddt',
        'plddt_cdr3a': 'cdr3a_plddt',
        'plddt_cdr1b': 'cdr1b_plddt',
        'plddt_cdr2b': 'cdr2b_plddt',
        'plddt_cdr3b': 'cdr3b_plddt',
        'pdockq_AB': 'pdockq',
        'iPSAE': 'ipsae'
    }
    
    for new_col, old_col in mapping.items():
        if new_col not in df_new.columns:
            if old_col in df_new.columns:
                df_new[new_col] = df_new[old_col]
    
    # Columnas promedio
    if 'avgipde' not in df_new.columns and all(c in df_new.columns for c in ['avgipde_a','avgipde_b']):
        df_new['avgipde'] = df_new[['avgipde_a','avgipde_b']].mean(axis=1)
    if 'avgipae' not in df_new.columns and all(c in df_new.columns for c in ['avgipae_a','avgipae_b']):
        df_new['avgipae'] = df_new[['avgipae_a','avgipae_b']].mean(axis=1)
    if 'avgpdockq2' not in df_new.columns and all(c in df_new.columns for c in ['pdockq2_a','pdockq2_b']):
        df_new['avgpdockq2'] = df_new[['pdockq2_a','pdockq2_b']].mean(axis=1)
    
    # Columnas obligatorias para RF
    features = [
        'global_plddt', 'plddt_cdr1a', 'plddt_cdr2a', 'plddt_cdr3a',
        'plddt_cdr1b', 'plddt_cdr2b', 'plddt_cdr3b',
        'iptm_mean', 'iptm_tcrpmhc', 'pdockq_AB',
        'avgipde', 'avgipae', 'avgpdockq2', 'iPSAE'
    ]
    
    missing = [f for f in features if f not in df_new.columns]
    if missing:
        raise KeyError(f"Faltan columnas necesarias para RF: {missing}")
    
    return df_new

def main():
    parser = argparse.ArgumentParser(description="""
            Train/Evaluate Random Forest model for TCR quality prediction.

            Cases:
            1) --test only           : Load pretrained model and make predictions on test set.
            2) --train (optionally --test) : Train RF model on training set and optionally predict on test set if not test provided predictions will be made in training set.
            """)
    
    parser.add_argument("--train", type=str, default=None, help="Path to training CSV file (optional)")
    parser.add_argument("--test", type=str, default=None, help="Path to test CSV file (optional)")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Optional per-class probability thresholds as JSON string, e.g. '{\"LQ\":0.4,\"AQ\":0.7,\"MQ\":0.7,\"HQ\":0.8}'")
    parser.add_argument("--save_model", type=str, default="../classifier/rf_model.pkl", help="Path to save trained RF model")
    args = parser.parse_args()

    # Parse thresholds if provided
    thresholds = None
    if args.thresholds is not None:
        import json
        thresholds = json.loads(args.thresholds)

    features = [
        'global_plddt', 'plddt_cdr1a', 'plddt_cdr2a', 'plddt_cdr3a',
        'plddt_cdr1b', 'plddt_cdr2b', 'plddt_cdr3b',
        'iptm_mean', 'iptm_tcrpmhc', 'pdockq_AB',
        'avgipde', 'avgipae', 'avgpdockq2', 'iPSAE'
    ]

    # CASE 1: Only test provided -> load pretrained model
    if args.train is None and args.test is not None:
        print("Loading pretrained model...")
        model = joblib.load("../classifier/rf_model_all.pkl")
        test_df = prepare_features(pd.read_csv(args.test))

        X_test = test_df[features]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        predictions = test_df.copy()
        predictions["predicted_label"] = y_pred
        proba_df = pd.DataFrame(
            y_proba,
            columns=[f"proba_{cls}" for cls in model.classes_],
            index=test_df.index
        )
        predictions = pd.concat([predictions, proba_df], axis=1)

        if thresholds is not None:
            classes = ['LQ','AQ','MQ','HQ']
            for cls in classes:
                predictions[f'pass_{cls}'] = (predictions[f'proba_{cls}'] >= thresholds[cls]).astype(int)
            def choose_label(row):
                passing = [cls for cls in classes if row[f'pass_{cls}']==1]
                if len(passing) == 0:
                    return 'LQ'
                probs = {cls: row[f'proba_{cls}'] for cls in passing}
                return max(probs, key=probs.get)
            predictions['predicted_label_thresholded'] = predictions.apply(choose_label, axis=1)
            predictions['pass_filter'] = predictions['predicted_label_thresholded'].isin(['AQ','MQ','HQ']).astype(int)

        predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        return

    # CASE 2: Train (with or without test)
    if args.train is not None:
        train_df = prepare_features(pd.read_csv(args.train))
        if args.test is not None:
            test_df = prepare_features(pd.read_csv(args.test))
        else:
            test_df = train_df.copy()

        model, predictions = train_rf(train_df, test_df, thresholds=thresholds)

        if args.save_model:
            joblib.dump(model, args.save_model)
            print(f"Trained model saved to {args.save_model}")

        predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()