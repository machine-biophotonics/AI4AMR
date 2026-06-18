import pandas as pd
dfs = {
    'matched': 'output_all_plates/all_plates_features_matched.csv',
    'zscore': 'output_all_plates/all_plates_features_zscore.csv',
}
for name, path in dfs.items():
    df = pd.read_csv(path)
    cols = sorted([c for c in df.columns if c not in ['plate','well','label','type','path']])
    print(f'=== {name} ===')
    print(f'  All: {cols}')
    print()
