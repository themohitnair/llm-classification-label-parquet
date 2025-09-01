import pandas as pd

# Load your labelled parquet
df = pd.read_parquet("temp_output.parquet")

total = len(df)
print(f"Total records: {total}")

# Specify which columns are labels. Update as needed!
label_columns = [col for col in df.columns if col not in ['description']]

for col in label_columns:
    print(f"\n{col.capitalize()} label distribution:")
    value_counts = df[col].value_counts(dropna=False).sort_index()
    percents = (value_counts / total * 100).round(2)
    for value, count in value_counts.items():
        print(f"{value}: {count} ({percents[value]}%)")
