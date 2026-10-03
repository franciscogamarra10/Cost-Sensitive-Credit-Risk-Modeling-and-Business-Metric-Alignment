"""Unified Variable Classification & Domain Feature Engineering."""
import pandas as pd
import numpy as np
from scipy import stats



def engineer_credit_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Calculates financial ratios and historical 6-month trends."""
    df = df_input.copy()

    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    if 'MARRIAGE' in df.columns:
        df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    pay_delay_cols = [c for c in df.columns if c.startswith('PAY_') and not c.startswith('PAY_AMT')]
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]

    if pay_delay_cols:
        payment_mapping = {-2: 0, -1: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}
        for col in pay_delay_cols:
            df[col] = df[col].replace(payment_mapping)

    for i in range(1, 7):
        b_col, p_col = f'BILL_AMT{i}', f'PAY_AMT{i}'
        if b_col in df.columns and 'LIMIT_BAL' in df.columns:
            df[f'UTIL_{i}'] = df[b_col] / (df['LIMIT_BAL'] + 1)
        if p_col in df.columns and b_col in df.columns:
            df[f'PAY_RATIO_{i}'] = df[p_col] / (df[b_col].abs() + 1)

    if bill_cols:
        df['AVG_BILL'] = df[bill_cols].mean(axis=1)
        df['STD_BILL'] = df[bill_cols].std(axis=1).fillna(0)
        df['MAX_BILL'] = df[bill_cols].max(axis=1)
        if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
            df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']

    if pay_amt_cols:
        df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)
        df['STD_PAY_AMT'] = df[pay_amt_cols].std(axis=1).fillna(0)
        if 'PAY_AMT1' in df.columns and 'PAY_AMT6' in df.columns:
            df['PAY_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']

    if pay_delay_cols:
        df['MAX_PAY_DELAY'] = df[pay_delay_cols].max(axis=1)
        df['AVG_PAY_DELAY'] = df[pay_delay_cols].mean(axis=1)
        df['COUNT_DELAY_GE2'] = (df[pay_delay_cols] >= 2).sum(axis=1)
        df['COUNT_DELAY_GT0'] = (df[pay_delay_cols] > 0).sum(axis=1)

    if 'BILL_AMT1' in df.columns and 'LIMIT_BAL' in df.columns:
        df['IS_OVER_LIMIT'] = (df['BILL_AMT1'] > df['LIMIT_BAL']).astype(int)

    return df
