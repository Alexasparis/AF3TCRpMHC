#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from itertools import combinations, product
from sklearn.metrics import precision_score, recall_score
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def create_nonoverlapping_folds(df, n_splits=5, random_state=42):
    """
    Create non-overlapping stratified folds by pdb_id.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'pdb_id' and 'Quality'.
    n_splits : int
        Number of folds to create.
    random_state : int
        Random seed.

    Returns
    -------
    folds : list of tuples
        List of (train_df, test_df) folds.
    """
    rng = np.random.default_rng(random_state)
    
    # Unique pdbs and their quality (take first occurrence)
    pdb_quality = df.drop_duplicates('pdb_id')[['pdb_id','Quality']].copy()

    # Shuffle pdbs
    pdb_quality = pdb_quality.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Stratify: split pdbs in roughly equal-sized folds, preserving quality distribution
    folds_pdbs = [[] for _ in range(n_splits)]
    
    for quality, group in pdb_quality.groupby('Quality'):
        pdbs = group['pdb_id'].tolist()
        for i, pdb in enumerate(pdbs):
            folds_pdbs[i % n_splits].append(pdb)

    all_pdbs = set(pdb_quality['pdb_id'])
    folds = []
    for i in range(n_splits):
        test_pdbs = set(folds_pdbs[i])
        train_pdbs = all_pdbs - test_pdbs

        train_df = df[df['pdb_id'].isin(train_pdbs)].copy()
        test_df = df[df['pdb_id'].isin(test_pdbs)].copy()

        folds.append((train_df, test_df))

    return folds

def explore_filters(folds_or_df, metrics, thresholds, target_col='Quality',
                    tiers_negative=['AQ'], tiers_positive=['MQ']):
    """
    Explore all combinations of metrics and logical operators (AND/OR)
    and calculate precision, recall, and FPR on the filtered subset by tiers.

    Parameters
    ----------
    folds_or_df : list of tuples or pd.DataFrame
        Either a list of (train_df, test_df) folds or a single DataFrame.
    metrics : list
        List of metric names to combine.
    thresholds : dict
        Dictionary {metric: [thresholds]} for binarization.
    target_col : str
        Column containing tier labels.
    tiers_negative : list of str
        Tiers considered negative (0).
    tiers_positive : list of str
        Tiers considered positive (1).
    """
    tier_map = {'LQ': 0, 'AQ': 1, 'MQ': 2, 'HQ': 3}

    # Flatten tiers to numbers
    neg_nums = [tier_map[t] for t in tiers_negative]
    pos_nums = [tier_map[t] for t in tiers_positive]

    # Generate all combinations of metrics and operators
    comb_list = []
    for r in range(1, len(metrics)+1):
        for comb in combinations(metrics, r):
            ops_list = [()] if len(comb) == 1 else list(product(['AND','OR'], repeat=len(comb)-1))
            for ops in ops_list:
                comb_list.append((comb, ops))
    print(f"Total combinations: {len(comb_list)}")

    results = []

    # Normalize folds
    if isinstance(folds_or_df, pd.DataFrame):
        folds = [(folds_or_df, None)]
    else:
        folds = folds_or_df

    for idx, (comb, ops) in enumerate(comb_list, 1):
        if idx % 1000 == 0 or idx == 1:
            print(f"Processing combination {idx}/{len(comb_list)}")
        precisions, recalls, fprs = [], [], []

        for train_df, _ in folds:
            df_eval = train_df.copy()
            df_eval['tier_num'] = df_eval[target_col].map(tier_map)
            df_eval = df_eval[df_eval['tier_num'].isin(neg_nums + pos_nums)]
            if df_eval.empty:
                continue

            # Binarize metrics
            binarized_arr = np.zeros((len(df_eval), len(metrics)), dtype=bool)
            for j, m in enumerate(metrics):
                thr = thresholds[m]
                # pick threshold based on tiers_positive (simplified: take max)
                idx_thr = 0 if len(thr)==2 else 1
                thr_val = thr[idx_thr]
                binarized_arr[:, j] = df_eval[m].values > thr_val

            # Apply metric combination
            comb_idx = [metrics.index(m) for m in comb]
            arr = binarized_arr[:, comb_idx]

            result = arr[:, 0].copy()
            for i, op in enumerate(ops):
                if op == 'AND':
                    result &= arr[:, i+1]
                else:
                    result |= arr[:, i+1]

            y_true = df_eval['tier_num'].isin(pos_nums).astype(int)
            y_pred = result.astype(int)

            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))

            FP = np.sum((y_pred==1) & (df_eval['tier_num'].isin(neg_nums)))
            TN = np.sum((y_pred==0) & (df_eval['tier_num'].isin(neg_nums)))
            fpr_fold = FP / (FP+TN) if (FP+TN)>0 else np.nan
            fprs.append(fpr_fold)

        results.append({
            'metrics': comb,
            'operators': ops,
            'precision_mean': np.mean(precisions) if precisions else np.nan,
            'recall_mean': np.mean(recalls) if recalls else np.nan,
            'false_positive_rate_mean': np.mean(fprs) if fprs else np.nan
        })

    return pd.DataFrame(results)

def evaluate_filter(folds_or_df, metrics, operators, thresholds, target_col='Quality',
                    tiers_negative=['AQ'], tiers_positive=['MQ']):
    """
    Evaluate a specific metric/operator filter on a DataFrame or multiple folds.

    Parameters
    ----------
    folds_or_df : pd.DataFrame or list of tuples
        Single DataFrame or list of (train_df, test_df) folds.
    metrics : list of str
        Metrics to use.
    operators : list of str
        Operators between metrics, length = len(metrics)-1
    thresholds : dict
        Dictionary {metric: [thresholds]} for binarization.
    target_col : str
        Column with tier labels.
    tiers_negative : list of str
        Negative tiers (0).
    tiers_positive : list of str
        Positive tiers (1).
    
    Returns
    -------
    dict
        Dictionary with mean precision, recall, FPR across folds.
    """
    tier_map = {'LQ':0, 'AQ':1, 'MQ':2, 'HQ':3}
    neg_nums = [tier_map[t] for t in tiers_negative]
    pos_nums = [tier_map[t] for t in tiers_positive]

    # Normalize folds
    if isinstance(folds_or_df, pd.DataFrame):
        folds = [(folds_or_df, None)]
    else:
        folds = folds_or_df

    precisions, recalls, fprs = [], [], []

    for train_df, _ in folds:
        df_eval = train_df.copy()
        df_eval['tier_num'] = df_eval[target_col].map(tier_map)
        df_eval = df_eval[df_eval['tier_num'].isin(neg_nums + pos_nums)]
        if df_eval.empty:
            continue

        # Binarize metrics
        binarized_arr = np.zeros((len(df_eval), len(metrics)), dtype=bool)
        for j, m in enumerate(metrics):
            thr = thresholds[m]
            # simple threshold: take middle if len==2 else max (you can adapt)
            idx_thr = 0 if len(thr)==2 else 1
            thr_val = thr[idx_thr]
            binarized_arr[:, j] = df_eval[m].values > thr_val

        # Apply operators
        result = binarized_arr[:, 0].copy()
        for i, op in enumerate(operators):
            if op == 'AND':
                result &= binarized_arr[:, i+1]
            else:
                result |= binarized_arr[:, i+1]

        y_true = df_eval['tier_num'].isin(pos_nums).astype(int)
        y_pred = result.astype(int)

        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

        FP = np.sum((y_pred==1) & (df_eval['tier_num'].isin(neg_nums)))
        TN = np.sum((y_pred==0) & (df_eval['tier_num'].isin(neg_nums)))
        fprs.append(FP / (FP+TN) if (FP+TN)>0 else np.nan)

    return {
        'precision_mean': np.mean(precisions) if precisions else np.nan,
        'recall_mean': np.mean(recalls) if recalls else np.nan,
        'false_positive_rate_mean': np.mean(fprs) if fprs else np.nan
    }

def assign_predicted_quality(
    df,
    metrics_list,       # list of metric tuples per filter: [metrics1, metrics2, metrics3]
    operators_list,     # list of operator tuples per filter: [ops1, ops2, ops3]
    splits=['LQ_vs_rest','LQ/AQ_vs_MQ/HQ','LQ/AQ/MQ_vs_HQ'],
):
    """
    Apply multiple superior-class filters and assign a predicted quality per row.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric columns and quality labels.
    metrics_list : list of tuples
        Metrics used for each filter, e.g. [metrics_lq, metrics_aq, metrics_mq]
    operators_list : list of tuples
        Logical operators for each filter, e.g. [ops_lq, ops_aq, ops_mq]
    splits : list of str
        Which tiers are considered superior per filter, in order of priority.
        Default ['LQ_vs_rest','LQ/AQ_vs_MQ/HQ','LQ/AQ/MQ_vs_HQ']
    target_col : str
        Column name for actual quality.

    Returns
    -------
    df : pd.DataFrame
        Original df with a new column 'quality_pred' containing predicted quality.
    """

    thresholds = {
        'plddt':[70,90],'cdr1b_plddt':[70,90],'cdr2a_plddt':[70,90],
        'cdr3a_plddt':[70,90],'cdr1b_plddt':[70,90],'cdr2b_plddt':[70,90],
        'cdr3b_plddt':[70,90],'iptm_mean':[0.6,0.8],'iptm_tcrpmhc':[0.6,0.8],
        'ipsae':[0.6,0.8],'avgpdockq2':[0.23,0.49,0.8],'pdockq':[0.23,0.49,0.8]
    }

    def binarize(series, metric):
        thr = thresholds[metric]
        if len(thr) == 2:
            return (series > thr[1]).astype(int)
        elif len(thr) == 3:
            return (series > thr[2]).astype(int)
        else:
            return (series > thr[0]).astype(int)

    def apply_logic(df_bin, ops):
        result = df_bin.iloc[:,0].copy()
        for i, op in enumerate(ops):
            col = df_bin.iloc[:, i+1]
            result = result & col if op=='AND' else result | col
        return result

    df_copy = df.copy()
    n_filters = len(metrics_list)
    
    # Apply each filter
    predictions_bin = []
    for i in range(n_filters):
        metrics = metrics_list[i]
        ops = operators_list[i]
        split = splits[i]
        df_bin = pd.DataFrame({m: binarize(df_copy[m], m) for m in metrics})
        pred = apply_logic(df_bin, ops)

        # Only assign 1 to superior tiers, else 0
        predictions_bin.append(pred.astype(int))

    # Combine predictions into a single quality_pred column
    quality_pred = pd.Series(["LQ"]*len(df_copy), index=df_copy.index)
    for pred, split in zip(predictions_bin[::-1], splits[::-1]):
        # Map split to quality label
        if split == 'LQ_vs_rest':
            q_label = 'AQ'
        elif split == 'LQ/AQ_vs_MQ/HQ':
            q_label = 'MQ'
        elif split == 'LQ/AQ/MQ_vs_HQ':
            q_label = 'HQ'
        else:
            q_label = 'LQ'
        # Assign label where pred==1 and not already HQ/MQ/AQ
        mask = (pred == 1) & (quality_pred == "LQ")
        quality_pred[mask] = q_label

    df_copy['quality_pred'] = quality_pred
    return df_copy

def main():

    parser = argparse.ArgumentParser(description="""Explore and evaluate metric filters for TCR structure quality classification.

Modes of operation:

1) --cv
   Explore all metric combinations using 5-fold cross-validation.
   Computes precision, recall, false positive rate, and harmonic mean.
   Outputs:
     - Filters CSV: ../filters/filter_<tiers_negative>_<tiers_positive>.csv
     - Evaluation CSV on test sets of the best filter: ../filters/eval_<tiers_negative>_<tiers_positive>.csv

2) --test
   Apply a predefined set of 3 filters to assign predicted quality (LQ, AQ, MQ, HQ) to each row.
   Requires --metrics and --operators specifying the filters and logic.
   Outputs:
     - Predicted qualities CSV: ../filters/predicted_qualities.csv""")

    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV containing metrics and 'Quality' column")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cv", action="store_true", help="Run filter exploration with 5-fold CV")
    mode.add_argument("--test", action="store_true", help="Test a tier classifier using 3 filters")

    parser.add_argument("--tiers_positive", nargs="+", default=["MQ","HQ"], help="Positive tiers for evaluation")
    parser.add_argument("--tiers_negative", nargs="+", default=["LQ","AQ"], help="Negative tiers for evaluation")

    parser.add_argument("--metrics", type=str, help="Example: \"[['plddt'],['iptm_mean','pdockq'],['avgpdockq2']]\"")
    parser.add_argument("--operators", type=str, help="Example: \"[['AND'],['OR']]\"")

    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)

    metrics_all=['plddt','cdr1a_plddt','cdr2a_plddt','cdr3a_plddt','cdr1b_plddt','cdr2b_plddt','cdr3b_plddt','iptm_mean','ipsae','iptm_tcrpmhc','pdockq','avgpdockq2']

    thresholds={'plddt':[70,90],'cdr1a_plddt':[70,90],'cdr2a_plddt':[70,90],'cdr3a_plddt':[70,90],'cdr1b_plddt':[70,90],'cdr2b_plddt':[70,90],'cdr3b_plddt':[70,90],'iptm_mean':[0.6,0.8],'ipsae':[0.6,0.8],'iptm_tcrpmhc':[0.6,0.8],'pdockq':[0.23,0.49],'avgpdockq2':[0.23,0.49,0.8]}

    if args.cv:
        folds=create_nonoverlapping_folds(df,n_splits=5)
        results_df=explore_filters(folds,metrics_all,thresholds,target_col="Quality",tiers_negative=args.tiers_negative,tiers_positive=args.tiers_positive)
        results_df["harmonic_mean"]=2*results_df["recall_mean"]*(1-results_df["false_positive_rate_mean"])/(results_df["recall_mean"]+(1-results_df["false_positive_rate_mean"]))
        results_sorted=results_df.sort_values("harmonic_mean",ascending=False).reset_index(drop=True)
        out=f"../filters/filter_{args.tiers_negative}_{args.tiers_positive}.csv"
        results_sorted.to_csv(out,index=False)

        print("\nTop filters:")
        print(results_sorted.head(5))

        best_filter=results_sorted.iloc[0]
        print("\nBest filter:",best_filter)

        result_dict = evaluate_filter(
            folds_or_df=folds,
            metrics=ast.literal_eval(best_filter['metrics']),
            operators=ast.literal_eval(best_filter['operators']),
            thresholds=thresholds,
            target_col='Quality',
            tiers_negative=args.tiers_negative,
            tiers_positive=args.tiers_positive
        )
        print("\nEvaluation of best filter in test sets:",result_dict)
        eval_df = pd.DataFrame([result_dict])
        out_eval = f"../filters/eval_{args.tiers_negative}_{args.tiers_positive}.csv"
        eval_df.to_csv(out_eval, index=False)
        print(f"Saved to {out_eval}")

    elif args.test:
        if args.metrics is None or args.operators is None:
            raise ValueError("For test mode you must provide --metrics and --operators")

        metrics_list=ast.literal_eval(args.metrics)
        operators_list=ast.literal_eval(args.operators)

        if len(metrics_list)!=3 or len(operators_list)!=3:
            raise ValueError("assign_predicted_quality requires 3 filters to assign LQ, AQ, MQ, HQ")

        preds_df=assign_predicted_quality(df,metrics_list,operators_list,splits=['LQ_vs_rest','LQ/AQ_vs_MQ/HQ','LQ/AQ/MQ_vs_HQ'])

        out="../filters/predicted_qualities.csv"
        preds_df.to_csv(out,index=False)
        print(f"\nPredictions saved to {out}")
        
if __name__ == "__main__":
    main()