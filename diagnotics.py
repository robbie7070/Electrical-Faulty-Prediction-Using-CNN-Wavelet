"""
Diagnostic script for CNN Wavelet Project
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH

# Convert to string if it's a Path object
DATA_PATH = str(DATA_PATH)

print("="*60)
print("🔍 PROJECT DIAGNOSTIC")
print("="*60)

# Check file extension
print(f"\n📁 DATA_PATH: {DATA_PATH}")
print(f"   File exists: {os.path.exists(DATA_PATH)}")

if not os.path.exists(DATA_PATH):
    print("❌ File not found!")
    sys.exit(1)

# Load based on file extension
if DATA_PATH.endswith('.csv'):
    df = pd.read_csv(DATA_PATH)
    print("✅ Loaded as CSV")
else:
    df = pd.read_excel(DATA_PATH, sheet_name='Dataset')
    print("✅ Loaded as Excel")

print(f"\n📊 Shape: {df.shape}")
print(f"📋 Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"   {i}: '{col}'")

print(f"\n📊 First 3 rows:")
print(df.head(3).to_string())

print(f"\n📊 Data types:")
print(df.dtypes)

# Check for key columns
checks = ['Sequence_ID', 'Timestep', 'Voltage_V', 'Current_A', 'Temperature_C', 'Fault_Type']
print(f"\n📊 Required columns check:")
for col in checks:
    if col in df.columns:
        print(f"   ✅ '{col}' found")
    else:
        print(f"   ❌ '{col}' MISSING!")

# Fault type analysis
if 'Fault_Type' in df.columns:
    print(f"\n📊 Fault Types:")
    print(df['Fault_Type'].value_counts())

# Sequence analysis
if 'Sequence_ID' in df.columns:
    print(f"\n📊 Sequences: {df['Sequence_ID'].nunique()}")
    seq_size = df.groupby('Sequence_ID').size()
    print(f"   Timesteps per sequence: {seq_size.iloc[0]} (min={seq_size.min()}, max={seq_size.max()})")

print("\n" + "="*60)
print("✅ Diagnostic complete!")
print("="*60)