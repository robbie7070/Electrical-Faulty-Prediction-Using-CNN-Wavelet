# data_loader.py
"""
Data loader module for Excel/CSV datasets with validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config


class DataLoader:
    """Load and validate power system fault data."""
    
    def __init__(self, file_path=None):
        self.file_path = Path(file_path) if file_path else config.DATA_PATH
        self.df = None
        self.sequences = None
        self.sequence_lengths = None
        
    def load_data(self):
        """Load data from Excel or CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        # Load based on file extension
        if self.file_path.suffix == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif self.file_path.suffix in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        print(f"✅ Loaded {len(self.df):,} records from {self.file_path.name}")
        return self.df
    
    def validate_columns(self):
        """Validate required columns exist."""
        required_columns = [
            config.SEQUENCE_ID_COL,
            config.TIMESTEP_COL,
            config.VOLTAGE_COL,
            config.CURRENT_COL,
            config.TEMPERATURE_COL,
            config.FAULT_TYPE_COL,
        ]
        
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("✅ All required columns present")
        
    def validate_data_types(self):
        """Validate and convert data types."""
        # Float columns
        for col in [config.VOLTAGE_COL, config.CURRENT_COL, config.TEMPERATURE_COL]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Integer columns
        for col in [config.SEQUENCE_ID_COL, config.TIMESTEP_COL]:
            if col in self.df.columns:
                # Keep Sequence_ID as string if it contains underscores
                if col == config.SEQUENCE_ID_COL and self.df[col].dtype == object:
                    pass  # Keep as string
                else:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print("✅ Data types validated")
        
    def check_missing_values(self):
        """Check for and report missing values."""
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️  Missing values detected:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count}")
            # Drop rows with missing values
            before = len(self.df)
            self.df = self.df.dropna()
            after = len(self.df)
            print(f"   Dropped {before - after} rows with missing values")
        else:
            print("✅ No missing values")
    
    def check_sequence_lengths(self):
        """Validate that all sequences have sufficient length."""
        self.sequence_lengths = self.df.groupby(config.SEQUENCE_ID_COL).size()
        
        total_seqs = len(self.sequence_lengths)
        valid_seqs = (self.sequence_lengths >= config.WINDOW_SIZE).sum()
        invalid_seqs = total_seqs - valid_seqs
        
        print(f"📊 Sequences: {total_seqs:,} total, {valid_seqs:,} valid (>= {config.WINDOW_SIZE} timesteps)")
        
        if invalid_seqs > 0:
            print(f"   Removing {invalid_seqs} short sequences...")
            valid_ids = self.sequence_lengths[self.sequence_lengths >= config.WINDOW_SIZE].index
            before = len(self.df)
            self.df = self.df[self.df[config.SEQUENCE_ID_COL].isin(valid_ids)]
            after = len(self.df)
            print(f"   Removed {before - after:,} records from short sequences")
        
        print(f"   Remaining: {len(self.df):,} records in {valid_seqs:,} sequences")
    
    def check_physics_consistency(self):
        """Check for physically impossible values - REPORT ONLY, don't drop."""
        issues = []
        
        # Voltage checks
        v_min = config.PHYSICS_CHECKS['voltage_absolute_min']
        v_max = config.PHYSICS_CHECKS['voltage_absolute_max']
        voltage_violations = (
            (self.df[config.VOLTAGE_COL] > v_max) |
            (self.df[config.VOLTAGE_COL] < v_min)
        )
        if voltage_violations.sum() > 0:
            issues.append(f"Voltage violations: {voltage_violations.sum():,} samples (range: {v_min}-{v_max}V)")
        
        # Current checks
        c_min = config.PHYSICS_CHECKS['current_absolute_min']
        c_max = config.PHYSICS_CHECKS['current_absolute_max']
        current_violations = (
            (self.df[config.CURRENT_COL] > c_max) |
            (self.df[config.CURRENT_COL] < c_min)
        )
        if current_violations.sum() > 0:
            issues.append(f"Current violations: {current_violations.sum():,} samples (range: {c_min}-{c_max}A)")
        
        # Temperature checks
        t_min = config.PHYSICS_CHECKS['temperature_absolute_min']
        t_max = config.PHYSICS_CHECKS['temperature_absolute_max']
        temp_violations = (
            (self.df[config.TEMPERATURE_COL] > t_max) |
            (self.df[config.TEMPERATURE_COL] < t_min)
        )
        if temp_violations.sum() > 0:
            issues.append(f"Temperature violations: {temp_violations.sum():,} samples (range: {t_min}-{t_max}°C)")
        
        if issues:
            print(f"⚠️  Physics violations detected (NOT dropping - adjust config if needed):")
            for issue in issues:
                print(f"   - {issue}")
            print(f"   💡 Update PHYSICS_CHECKS in config.py to match your data ranges")
        else:
            print("✅ All physics checks passed")
    
    def get_sequences(self):
        """Group data by sequence ID."""
        self.sequences = {
            seq_id: group.sort_values(config.TIMESTEP_COL)
            for seq_id, group in self.df.groupby(config.SEQUENCE_ID_COL)
        }
        print(f"✅ Extracted {len(self.sequences):,} sequences")
        return self.sequences
    
    def get_statistics(self):
        """Get dataset statistics."""
        stats = {
            'total_records': len(self.df),
            'total_sequences': len(self.sequences) if self.sequences else 0,
            'voltage_stats': {
                'mean': self.df[config.VOLTAGE_COL].mean(),
                'std': self.df[config.VOLTAGE_COL].std(),
                'min': self.df[config.VOLTAGE_COL].min(),
                'max': self.df[config.VOLTAGE_COL].max(),
            },
            'current_stats': {
                'mean': self.df[config.CURRENT_COL].mean(),
                'std': self.df[config.CURRENT_COL].std(),
                'min': self.df[config.CURRENT_COL].min(),
                'max': self.df[config.CURRENT_COL].max(),
            },
            'temperature_stats': {
                'mean': self.df[config.TEMPERATURE_COL].mean(),
                'std': self.df[config.TEMPERATURE_COL].std(),
                'min': self.df[config.TEMPERATURE_COL].min(),
                'max': self.df[config.TEMPERATURE_COL].max(),
            },
            'fault_distribution': self.df[config.FAULT_TYPE_COL].value_counts().to_dict(),
        }
        return stats
    
    def run_full_validation(self):
        """Run complete validation pipeline."""
        print("\n" + "="*60)
        print("🔍 RUNNING DATA VALIDATION")
        print("="*60)
        
        self.load_data()
        self.validate_columns()
        self.validate_data_types()
        self.check_missing_values()
        self.check_physics_consistency()
        self.check_sequence_lengths()
        self.get_sequences()
        
        stats = self.get_statistics()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Records: {stats['total_records']:,}")
        print(f"   Sequences: {stats['total_sequences']:,}")
        print(f"   Voltage: {stats['voltage_stats']['mean']:.1f}V ± {stats['voltage_stats']['std']:.1f}V")
        print(f"   Current: {stats['current_stats']['mean']:.1f}A ± {stats['current_stats']['std']:.1f}A")
        print(f"   Temperature: {stats['temperature_stats']['mean']:.1f}°C ± {stats['temperature_stats']['std']:.1f}°C")
        
        print(f"\n📊 Fault Distribution:")
        for fault, count in stats['fault_distribution'].items():
            pct = count / stats['total_records'] * 100
            print(f"   {fault}: {count:,} ({pct:.1f}%)")
        
        print("="*60 + "\n")
        return self.df, self.sequences