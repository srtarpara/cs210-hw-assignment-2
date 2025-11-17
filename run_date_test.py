import pandas as pd
import numpy as np


def normalize_date_columns(df_input, date_cols):
    df_local = df_input.copy()

    for col in date_cols:
        if col in df_local.columns:
            # Parse element-wise to allow mixed formats per-row
            parsed = df_local[col].apply(lambda x: pd.to_datetime(x, errors='coerce'))

            formatted = parsed.dt.strftime('%Y-%m-%d')
            formatted = formatted.where(parsed.notna(), pd.NA)
            df_local[col] = formatted
        else:
            print(f"Column '{col}' not found in dataset.")

    return df_local


example_inv = pd.DataFrame({
    "Date of Admission": ["06-19-2025", "2025-11-17", "10.23.2005", "12/31/1999"],
    "Discharge Date": ["2025/07/01", "11-25-2025", "2005.11.01", "01/15/2000"]
})

print("Example BEFORE:")
print(example_inv)

print("\nParsing each individual value using pd.to_datetime(..., errors='coerce'):")
for col in example_inv.columns:
    print(f"Column: {col}")
    for val in example_inv[col].tolist():
        parsed_single = pd.to_datetime(val, errors='coerce')
        print(f"  original: {val!r} -> parsed: {parsed_single!r}")

example_clean = normalize_date_columns(example_inv, ["Date of Admission", "Discharge Date"])

print("\nExample AFTER:")
print(example_clean)

print("\nDtypes:")
print(example_clean.dtypes)

# show underlying values including NaT check
print("\nUnderlying values (repr):")
for col in example_clean.columns:
    print(col, [repr(x) for x in example_clean[col].tolist()])
