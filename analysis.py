"""
================================================================================
AADHAAR SOCIETAL TRENDS ANALYSIS (COMPLETE)
================================================================================
Problem Statement: Unlocking Societal Trends in Aadhaar Enrolment and Updates

This analysis identifies meaningful patterns, trends, anomalies, and predictive 
indicators from UIDAI's Aadhaar enrolment and update datasets to support 
informed decision-making and system improvements.

Includes: 
1. Rigorous State Name Standardization (Cleaning)
2. Advanced Visualizations (Univariate, Bivariate, Trivariate)
3. Anomaly Detection
4. Executive Summary Generation

Author: Data Analysis Team
Date: January 2026
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import os

# Configure display settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ----------------------------------------------------------------------------
# PATH CONFIGURATION
# Update these paths to match your folder structure if different
# ----------------------------------------------------------------------------
BASE_PATH = Path(r"c:\Users\thevi\OneDrive\Desktop\aadhar-hackathon")
BIOMETRIC_PATH = BASE_PATH / "aadhar_biometric"
DEMOGRAPHIC_PATH = BASE_PATH / "aadhar_demographic"
ENROLMENT_PATH = BASE_PATH / "aadhar_enrolment"
OUTPUT_PATH = BASE_PATH / "output"

# Create output directory
OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("AADHAAR SOCIETAL TRENDS ANALYSIS")
print("=" * 80)
print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Base Path: {BASE_PATH}")

# ============================================================================
# SECTION 2: DATA LOADING & MAPPING CONSTANTS
# ============================================================================

# ----------------------------------------------------------------------------
# STATE NAME MAPPING (Rigorous Standardization)
# ----------------------------------------------------------------------------
STATE_NAME_MAPPING = {
    # Standard names (no change needed)
    'Andhra Pradesh': 'Andhra Pradesh',
    'Arunachal Pradesh': 'Arunachal Pradesh',
    'Assam': 'Assam',
    'Bihar': 'Bihar',
    'Chandigarh': 'Chandigarh',
    'Delhi': 'Delhi',
    'Goa': 'Goa',
    'Gujarat': 'Gujarat',
    'Haryana': 'Haryana',
    'Himachal Pradesh': 'Himachal Pradesh',
    'Jharkhand': 'Jharkhand',
    'Karnataka': 'Karnataka',
    'Kerala': 'Kerala',
    'Ladakh': 'Ladakh',
    'Lakshadweep': 'Lakshadweep',
    'Madhya Pradesh': 'Madhya Pradesh',
    'Maharashtra': 'Maharashtra',
    'Manipur': 'Manipur',
    'Meghalaya': 'Meghalaya',
    'Mizoram': 'Mizoram',
    'Nagaland': 'Nagaland',
    'Punjab': 'Punjab',
    'Rajasthan': 'Rajasthan',
    'Sikkim': 'Sikkim',
    'Tamil Nadu': 'Tamil Nadu',
    'Telangana': 'Telangana',
    'Tripura': 'Tripura',
    'Uttar Pradesh': 'Uttar Pradesh',
    'Uttarakhand': 'Uttarakhand',
    
    # Andaman and Nicobar Islands variations
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Andaman and Nicobar Islands': 'Andaman and Nicobar Islands',
    
    # Andhra Pradesh variations (lowercase)
    'andhra pradesh': 'Andhra Pradesh',
    
    # Chhattisgarh variations
    'Chhattisgarh': 'Chhattisgarh',
    'Chhatisgarh': 'Chhattisgarh',  # Missing 't'
    
    # Dadra and Nagar Haveli variations
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra and Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra and Nagar Haveli and Daman and Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    
    # Daman and Diu variations
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman and Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    
    # Jammu and Kashmir variations
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Jammu And Kashmir': 'Jammu and Kashmir',
    'Jammu and Kashmir': 'Jammu and Kashmir',
    
    # Odisha variations
    'Odisha': 'Odisha',
    'ODISHA': 'Odisha',
    'Orissa': 'Odisha',  # Old name
    'odisha': 'Odisha',  # Lowercase
    
    # Puducherry variations
    'Puducherry': 'Puducherry',
    'Pondicherry': 'Puducherry',  # Old name
    
    # Tamil Nadu variations
    'Tamilnadu': 'Tamil Nadu',  # No space
    
    # Uttarakhand variations
    'Uttaranchal': 'Uttarakhand',  # Old name
    
    # West Bengal variations
    'West Bengal': 'West Bengal',
    'WEST BENGAL': 'West Bengal',
    'WESTBENGAL': 'West Bengal',
    'West  Bengal': 'West Bengal',
    'West Bangal': 'West Bengal',
    'West Bengli': 'West Bengal',
    'West bengal': 'West Bengal',
    'Westbengal': 'West Bengal',
    'west Bengal': 'West Bengal',
    
    # ERRONEOUS ENTRIES - Mapped to correct states
    '100000': 'INVALID_ENTRY',
    'BALANAGAR': 'INVALID_ENTRY',
    'Darbhanga': 'Bihar',  
    'Jaipur': 'Rajasthan', 
    'Madanapalle': 'Andhra Pradesh',
    'Nagpur': 'Maharashtra', 
    'Puttenahalli': 'Karnataka', 
    'Raja Annamalai Puram': 'Tamil Nadu',
}

def load_dataset(folder_path, dataset_name):
    """Load all CSV files from a folder and concatenate them."""
    all_files = list(folder_path.glob("*.csv"))
    print(f"\nLoading {dataset_name} dataset...")
    print(f"  Found {len(all_files)} files")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"    - Loaded: {file.name}")
        except Exception as e:
            print(f"    ! Error loading {file.name}: {e}")
            
    if not dfs:
        print("    ! No files loaded.")
        return pd.DataFrame()
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Total records: {len(combined_df):,}")
    return combined_df

# Load all datasets
df_biometric = load_dataset(BIOMETRIC_PATH, "Biometric Updates")
df_demographic = load_dataset(DEMOGRAPHIC_PATH, "Demographic Updates")
df_enrolment = load_dataset(ENROLMENT_PATH, "Enrolment")

# ============================================================================
# SECTION 3: DATA PREPROCESSING (CLEANING & STANDARDIZATION)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: DATA PREPROCESSING")
print("=" * 80)

def preprocess_and_standardize(df, name):
    """
    Apply rigorous state mapping, remove invalid entries, and clean dates.
    """
    df = df.copy()
    initial_count = len(df)
    
    # 1. Standardize State Names
    df['state_original'] = df['state']
    df['state'] = df['state'].map(STATE_NAME_MAPPING)
    
    # 2. Remove Invalid/Unmapped Entries
    df = df[df['state'].notna()]  # Remove unmapped
    df = df[df['state'] != 'INVALID_ENTRY'] # Remove specific invalid codes
    
    # 3. Clean District Names
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.strip()
    
    # 4. Parse Dates (dd-mm-yyyy format)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # 5. Feature Engineering (Time)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    removed = initial_count - len(df)
    print(f"[{name}] Cleaned. Removed {removed} invalid records. Final count: {len(df):,}")
    return df

# Define Zone Mapping for Regional Analysis
ZONE_MAPPING = {
    'North': ['Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Ladakh', 
              'Punjab', 'Rajasthan', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh'],
    'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 
              'Puducherry', 'Lakshadweep', 'Andaman and Nicobar Islands'],
    'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal'],
    'West': ['Goa', 'Gujarat', 'Maharashtra', 'Dadra and Nagar Haveli and Daman and Diu'],
    'Central': ['Chhattisgarh', 'Madhya Pradesh'],
    'Northeast': ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 
                  'Nagaland', 'Sikkim', 'Tripura']
}

def add_zone(df):
    """Add zone column based on state."""
    zone_lookup = {}
    for zone, states in ZONE_MAPPING.items():
        for state in states:
            zone_lookup[state] = zone
    df['zone'] = df['state'].map(zone_lookup).fillna('Other')
    return df

# Apply Cleaning
print("\nApplying standardization logic...")
df_biometric = preprocess_and_standardize(df_biometric, "Biometric")
df_demographic = preprocess_and_standardize(df_demographic, "Demographic")
df_enrolment = preprocess_and_standardize(df_enrolment, "Enrolment")

# Calculate Total Columns
df_biometric['total_biometric'] = df_biometric['bio_age_5_17'] + df_biometric['bio_age_17_']
df_demographic['total_demographic'] = df_demographic['demo_age_5_17'] + df_demographic['demo_age_17_']
df_enrolment['total_enrolment'] = df_enrolment['age_0_5'] + df_enrolment['age_5_17'] + df_enrolment['age_18_greater']

# Apply Zoning
df_biometric = add_zone(df_biometric)
df_demographic = add_zone(df_demographic)
df_enrolment = add_zone(df_enrolment)

# Save Intermediate Cleaned Files
print("\nSaving cleaned datasets to output folder...")
df_biometric.to_csv(OUTPUT_PATH / 'cleaned_biometric.csv', index=False)
df_demographic.to_csv(OUTPUT_PATH / 'cleaned_demographic.csv', index=False)
df_enrolment.to_csv(OUTPUT_PATH / 'cleaned_enrolment.csv', index=False)

# ============================================================================
# SECTION 4: DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: DATA QUALITY ASSESSMENT")
print("=" * 80)

def assess_data_quality(df, name):
    print(f"\n{name} Dataset Quality:")
    print(f"  Total Records: {len(df):,}")
    print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique States: {df['state'].nunique()}")

assess_data_quality(df_biometric, "Biometric Updates")
assess_data_quality(df_demographic, "Demographic Updates")
assess_data_quality(df_enrolment, "Enrolment")

# ============================================================================
# SECTION 5: UNIVARIATE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: UNIVARIATE ANALYSIS")
print("=" * 80)

# Create figure for univariate analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Univariate Analysis: Distribution of Aadhaar Activities', fontsize=16, fontweight='bold')

# 5.1 Age Group Distribution - Enrolment
ax1 = axes[0, 0]
enrol_age_totals = [
    df_enrolment['age_0_5'].sum(),
    df_enrolment['age_5_17'].sum(),
    df_enrolment['age_18_greater'].sum()
]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax1.bar(['0-5 Years', '5-17 Years', '18+ Years'], enrol_age_totals, color=colors)
ax1.set_title('Enrolment by Age Group', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Enrolments')
for bar, val in zip(bars, enrol_age_totals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val/1e6:.2f}M', 
             ha='center', va='bottom', fontsize=10)

# 5.2 Age Group Distribution - Biometric Updates
ax2 = axes[0, 1]
bio_age_totals = [
    df_biometric['bio_age_5_17'].sum(),
    df_biometric['bio_age_17_'].sum()
]
bars = ax2.bar(['5-17 Years', '17+ Years'], bio_age_totals, color=['#4ECDC4', '#45B7D1'])
ax2.set_title('Biometric Updates by Age Group', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Updates')
for bar, val in zip(bars, bio_age_totals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val/1e6:.2f}M', 
             ha='center', va='bottom', fontsize=10)

# 5.3 Age Group Distribution - Demographic Updates
ax3 = axes[0, 2]
demo_age_totals = [
    df_demographic['demo_age_5_17'].sum(),
    df_demographic['demo_age_17_'].sum()
]
bars = ax3.bar(['5-17 Years', '17+ Years'], demo_age_totals, color=['#4ECDC4', '#45B7D1'])
ax3.set_title('Demographic Updates by Age Group', fontsize=12, fontweight='bold')
ax3.set_ylabel('Total Updates')
for bar, val in zip(bars, demo_age_totals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val/1e6:.2f}M', 
             ha='center', va='bottom', fontsize=10)

# 5.4 Zone-wise Distribution - All Activities
ax4 = axes[1, 0]
zone_data = pd.DataFrame({
    'Enrolment': df_enrolment.groupby('zone')['total_enrolment'].sum(),
    'Biometric': df_biometric.groupby('zone')['total_biometric'].sum(),
    'Demographic': df_demographic.groupby('zone')['total_demographic'].sum()
})
zone_data.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax4.set_title('Activity by Zone', fontsize=12, fontweight='bold')
ax4.set_xlabel('Zone')
ax4.set_ylabel('Total Count')
ax4.tick_params(axis='x', rotation=45)
ax4.legend(title='Activity Type')

# 5.5 Top 10 States by Enrolment
ax5 = axes[1, 1]
top_states_enrol = df_enrolment.groupby('state')['total_enrolment'].sum().nlargest(10)
bars = ax5.barh(range(len(top_states_enrol)), top_states_enrol.values, color='#FF6B6B')
ax5.set_yticks(range(len(top_states_enrol)))
ax5.set_yticklabels(top_states_enrol.index)
ax5.set_title('Top 10 States by Enrolment', fontsize=12, fontweight='bold')
ax5.set_xlabel('Total Enrolments')
ax5.invert_yaxis()

# 5.6 Activity Type Comparison
ax6 = axes[1, 2]
activity_totals = {
    'New\nEnrolments': df_enrolment['total_enrolment'].sum(),
    'Biometric\nUpdates': df_biometric['total_biometric'].sum(),
    'Demographic\nUpdates': df_demographic['total_demographic'].sum()
}
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
wedges, texts, autotexts = ax6.pie(activity_totals.values(), labels=activity_totals.keys(),
                                    autopct='%1.1f%%', colors=colors, startangle=90)
ax6.set_title('Overall Activity Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'univariate_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: univariate_analysis.png")

# ============================================================================
# SECTION 6: BIVARIATE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: BIVARIATE ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Bivariate Analysis: Relationships in Aadhaar Data', fontsize=16, fontweight='bold')

# 6.1 State-wise Heatmap - Enrolment by Age Group
ax1 = axes[0, 0]
state_age_enrol = df_enrolment.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
top_15_states = state_age_enrol.sum(axis=1).nlargest(15).index
state_age_enrol_top = state_age_enrol.loc[top_15_states]
sns.heatmap(state_age_enrol_top, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1, 
            cbar_kws={'label': 'Count'})
ax1.set_title('Top 15 States: Enrolment by Age Group', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age Group')
ax1.set_ylabel('State')

# 6.2 Biometric vs Demographic Updates by Zone
ax2 = axes[0, 1]
zone_comparison = pd.DataFrame({
    'Biometric': df_biometric.groupby('zone')['total_biometric'].sum(),
    'Demographic': df_demographic.groupby('zone')['total_demographic'].sum()
})
zone_comparison.plot(kind='bar', ax=ax2, color=['#4ECDC4', '#45B7D1'], width=0.8)
ax2.set_title('Biometric vs Demographic Updates by Zone', fontsize=12, fontweight='bold')
ax2.set_xlabel('Zone')
ax2.set_ylabel('Total Updates')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Update Type')

# 6.3 Scatter: Youth vs Adult Updates (Biometric)
ax3 = axes[1, 0]
state_bio = df_biometric.groupby('state')[['bio_age_5_17', 'bio_age_17_']].sum()
ax3.scatter(state_bio['bio_age_5_17'], state_bio['bio_age_17_'], alpha=0.6, s=100, c='#FF6B6B')
ax3.set_xlabel('Youth (5-17) Biometric Updates')
ax3.set_ylabel('Adult (17+) Biometric Updates')
ax3.set_title('Youth vs Adult Biometric Updates by State', fontsize=12, fontweight='bold')

# Add correlation
if len(state_bio) > 1:
    corr = state_bio['bio_age_5_17'].corr(state_bio['bio_age_17_'])
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Add trend line
    z = np.polyfit(state_bio['bio_age_5_17'], state_bio['bio_age_17_'], 1)
    p = np.poly1d(z)
    ax3.plot(state_bio['bio_age_5_17'].sort_values(), 
             p(state_bio['bio_age_5_17'].sort_values()), "r--", alpha=0.8, label='Trend Line')
    ax3.legend()

# 6.4 Child Enrolment Analysis (0-5 vs 5-17)
ax4 = axes[1, 1]
state_child = df_enrolment.groupby('state')[['age_0_5', 'age_5_17']].sum()
ax4.scatter(state_child['age_0_5'], state_child['age_5_17'], alpha=0.6, s=100, c='#45B7D1')
ax4.set_xlabel('Infant/Toddler Enrolments (0-5)')
ax4.set_ylabel('Youth Enrolments (5-17)')
ax4.set_title('Child Enrolment Patterns by State', fontsize=12, fontweight='bold')

if len(state_child) > 1:
    corr_child = state_child['age_0_5'].corr(state_child['age_5_17'])
    ax4.text(0.05, 0.95, f'Correlation: {corr_child:.3f}', transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'bivariate_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: bivariate_analysis.png")

# ============================================================================
# SECTION 7: TEMPORAL ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 7: TEMPORAL ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Temporal Analysis: Time-based Patterns', fontsize=16, fontweight='bold')

# 7.1 Daily Activity Trends - Enrolment
ax1 = axes[0, 0]
daily_enrol = df_enrolment.groupby('date')['total_enrolment'].sum().reset_index()
if not daily_enrol.empty:
    ax1.plot(daily_enrol['date'], daily_enrol['total_enrolment'], color='#FF6B6B', linewidth=1.5)
    ax1.fill_between(daily_enrol['date'], daily_enrol['total_enrolment'], alpha=0.3, color='#FF6B6B')
ax1.set_title('Daily Enrolment Trends', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Enrolments')
ax1.tick_params(axis='x', rotation=45)

# 7.2 Day of Week Analysis
ax2 = axes[0, 1]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_enrol = df_enrolment.groupby('day_of_week')['total_enrolment'].mean().reindex(day_order)
dow_bio = df_biometric.groupby('day_of_week')['total_biometric'].mean().reindex(day_order)
dow_demo = df_demographic.groupby('day_of_week')['total_demographic'].mean().reindex(day_order)

x = np.arange(len(day_order))
width = 0.25
ax2.bar(x - width, dow_enrol, width, label='Enrolment', color='#FF6B6B')
ax2.bar(x, dow_bio, width, label='Biometric', color='#4ECDC4')
ax2.bar(x + width, dow_demo, width, label='Demographic', color='#45B7D1')
ax2.set_xticks(x)
ax2.set_xticklabels(day_order, rotation=45, ha='right')
ax2.set_title('Average Activity by Day of Week', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Count per Record')
ax2.legend()

# 7.3 Monthly Distribution
ax3 = axes[1, 0]
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_enrol = df_enrolment.groupby('month_name')['total_enrolment'].sum()
available_months = [m for m in month_order if m in month_enrol.index]
if available_months:
    month_enrol = month_enrol.reindex(available_months)
    bars = ax3.bar(available_months, month_enrol.values, color='#FF6B6B')
    ax3.set_title('Monthly Enrolment Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Total Enrolments')
    ax3.tick_params(axis='x', rotation=45)

# 7.4 Week of Year Trends
ax4 = axes[1, 1]
week_enrol = df_enrolment.groupby('week_of_year')['total_enrolment'].sum().sort_index()
if not week_enrol.empty:
    ax4.plot(week_enrol.index, week_enrol.values, color='#4ECDC4', linewidth=2, marker='o', markersize=4)
    ax4.fill_between(week_enrol.index, week_enrol.values, alpha=0.3, color='#4ECDC4')
ax4.set_title('Weekly Enrolment Patterns', fontsize=12, fontweight='bold')
ax4.set_xlabel('Week of Year')
ax4.set_ylabel('Total Enrolments')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: temporal_analysis.png")

# ============================================================================
# SECTION 8: GEOGRAPHIC ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 8: GEOGRAPHIC ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Geographic Analysis: Regional Patterns', fontsize=16, fontweight='bold')

# 8.1 State-wise Activity Comparison
ax1 = axes[0, 0]
state_summary = pd.DataFrame({
    'Enrolment': df_enrolment.groupby('state')['total_enrolment'].sum(),
    'Biometric': df_biometric.groupby('state')['total_biometric'].sum(),
    'Demographic': df_demographic.groupby('state')['total_demographic'].sum()
}).fillna(0)
state_summary['Total'] = state_summary.sum(axis=1)
top_10_total = state_summary.nlargest(10, 'Total')

top_10_total[['Enrolment', 'Biometric', 'Demographic']].plot(kind='barh', ax=ax1, 
                                                              color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                              stacked=True)
ax1.set_title('Top 10 States: Total Aadhaar Activity', fontsize=12, fontweight='bold')
ax1.set_xlabel('Total Count')
ax1.set_ylabel('State')
ax1.legend(title='Activity Type', loc='lower right')

# 8.2 Zone-wise Breakdown
ax2 = axes[0, 1]
zone_totals = zone_data.sum(axis=1)
colors = plt.cm.Set3(np.linspace(0, 1, len(zone_totals)))
wedges, texts, autotexts = ax2.pie(zone_totals, labels=zone_totals.index, autopct='%1.1f%%',
                                     colors=colors, startangle=90, explode=[0.02]*len(zone_totals))
ax2.set_title('Activity Distribution by Zone', fontsize=12, fontweight='bold')

# 8.3 District Density - Top Performers
ax3 = axes[1, 0]
district_enrol = df_enrolment.groupby(['state', 'district'])['total_enrolment'].sum().reset_index()
top_districts = district_enrol.nlargest(15, 'total_enrolment')
bars = ax3.barh(top_districts['district'] + '\n(' + top_districts['state'].str[:10] + ')', 
                top_districts['total_enrolment'], color='#FF6B6B')
ax3.set_title('Top 15 Districts by Enrolment', fontsize=12, fontweight='bold')
ax3.set_xlabel('Total Enrolments')
ax3.invert_yaxis()

# 8.4 Bottom Performers
ax4 = axes[1, 1]
bottom_10_states = state_summary.nsmallest(10, 'Total')
bottom_10_states[['Enrolment', 'Biometric', 'Demographic']].plot(kind='barh', ax=ax4, 
                                                                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                                                                  stacked=True)
ax4.set_title('Bottom 10 States: Areas Needing Attention', fontsize=12, fontweight='bold')
ax4.set_xlabel('Total Count')
ax4.set_ylabel('State')
ax4.legend(title='Activity Type', loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'geographic_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: geographic_analysis.png")

# ============================================================================
# SECTION 9: ANOMALY DETECTION
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 9: ANOMALY DETECTION")
print("=" * 80)

def detect_anomalies_iqr(series, name):
    """Detect anomalies using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = series[(series < lower_bound) | (series > upper_bound)]
    print(f"\n{name}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    print(f"  Anomalies found: {len(anomalies)} ({100*len(anomalies)/len(series):.2f}%)")
    
    return anomalies, lower_bound, upper_bound

# Detect anomalies in state-level totals
state_enrol_totals = df_enrolment.groupby('state')['total_enrolment'].sum()
state_bio_totals = df_biometric.groupby('state')['total_biometric'].sum()
state_demo_totals = df_demographic.groupby('state')['total_demographic'].sum()

anomalies_enrol, lb_e, ub_e = detect_anomalies_iqr(state_enrol_totals, "State Enrolment Totals")
anomalies_bio, lb_b, ub_b = detect_anomalies_iqr(state_bio_totals, "State Biometric Totals")
anomalies_demo, lb_d, ub_d = detect_anomalies_iqr(state_demo_totals, "State Demographic Totals")

# Visualize anomalies
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Anomaly Detection: State-Level Activity', fontsize=16, fontweight='bold')

for ax, data, anomalies, title, color in [
    (axes[0], state_enrol_totals, anomalies_enrol, 'Enrolment', '#FF6B6B'),
    (axes[1], state_bio_totals, anomalies_bio, 'Biometric Updates', '#4ECDC4'),
    (axes[2], state_demo_totals, anomalies_demo, 'Demographic Updates', '#45B7D1')
]:
    ax.boxplot(data, vert=True, patch_artist=True, 
               boxprops=dict(facecolor=color, alpha=0.6))
    if not anomalies.empty:
        ax.scatter([1]*len(anomalies), anomalies, color='red', s=100, zorder=5, label='Anomalies')
    ax.set_title(f'{title}\n({len(anomalies)} anomalies)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Count')
    
    # Annotate top anomalies
    for state, value in anomalies.nlargest(3).items():
        ax.annotate(state[:15], (1.1, value), fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'anomaly_detection.png', dpi=300, bbox_inches='tight')
print("Saved: anomaly_detection.png")

# ============================================================================
# SECTION 10: TRIVARIATE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 10: TRIVARIATE ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Trivariate Analysis: Multi-dimensional Insights', fontsize=16, fontweight='bold')

# 10.1 Zone x Age Group x Activity Type Heatmap
ax1 = axes[0]
zone_age_data = pd.DataFrame({
    'Zone': list(ZONE_MAPPING.keys()) * 3,
    'Age_Group': ['Youth'] * len(ZONE_MAPPING) + ['Adult'] * len(ZONE_MAPPING) + ['Infant'] * len(ZONE_MAPPING),
    'Count': [df_enrolment[df_enrolment['zone'] == z]['age_5_17'].sum() for z in ZONE_MAPPING] +
             [df_enrolment[df_enrolment['zone'] == z]['age_18_greater'].sum() for z in ZONE_MAPPING] +
             [df_enrolment[df_enrolment['zone'] == z]['age_0_5'].sum() for z in ZONE_MAPPING]
})
pivot_data = zone_age_data.pivot(index='Zone', columns='Age_Group', values='Count')
sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax1,
            cbar_kws={'label': 'Enrolment Count'})
ax1.set_title('Zone × Age Group Enrolment Matrix', fontsize=12, fontweight='bold')

# 10.2 3D-like bubble chart: State, Update Ratio, Volume
ax2 = axes[1]
state_combined = pd.DataFrame({
    'Biometric': state_bio_totals,
    'Demographic': state_demo_totals,
    'Enrolment': state_enrol_totals
}).fillna(0)
state_combined['Total'] = state_combined.sum(axis=1)
state_combined['Bio_Ratio'] = state_combined['Biometric'] / state_combined['Total']
state_combined['Demo_Ratio'] = state_combined['Demographic'] / state_combined['Total']

# Filter to top 20 states by volume
top_20 = state_combined.nlargest(20, 'Total')
scatter = ax2.scatter(top_20['Bio_Ratio'], top_20['Demo_Ratio'], 
                        s=top_20['Total']/10000, alpha=0.6, c=top_20['Enrolment'], 
                        cmap='viridis')
ax2.set_xlabel('Biometric Update Ratio')
ax2.set_ylabel('Demographic Update Ratio')
ax2.set_title('State Activity Profile (bubble size = total volume)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Enrolment Count')

# Annotate some points
for state in top_20.nlargest(5, 'Total').index:
    ax2.annotate(state[:10], (top_20.loc[state, 'Bio_Ratio'], top_20.loc[state, 'Demo_Ratio']),
                 fontsize=8, alpha=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'trivariate_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: trivariate_analysis.png")

# ============================================================================
# SECTION 11: KEY STATISTICS AND INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 11: KEY STATISTICS AND INSIGHTS")
print("=" * 80)

total_enrolments = df_enrolment['total_enrolment'].sum()
total_biometric = df_biometric['total_biometric'].sum()
total_demographic = df_demographic['total_demographic'].sum()
grand_total = total_enrolments + total_biometric + total_demographic

print(f"\nTotal Aadhaar Activities: {grand_total:,.0f}")
print(f"  - New Enrolments: {total_enrolments:,.0f} ({100*total_enrolments/grand_total if grand_total else 0:.1f}%)")
print(f"  - Biometric Updates: {total_biometric:,.0f} ({100*total_biometric/grand_total if grand_total else 0:.1f}%)")
print(f"  - Demographic Updates: {total_demographic:,.0f} ({100*total_demographic/grand_total if grand_total else 0:.1f}%)")

print("\n" + "-" * 60)
print("KEY INSIGHTS")
print("-" * 60)

if total_enrolments > 0:
    insights = [
        f"1. DEMOGRAPHIC DOMINANCE: Demographic updates ({100*total_demographic/grand_total:.1f}%) dominate, indicating frequent address/name changes.",
        f"2. ADULT FOCUS: {100*df_enrolment['age_18_greater'].sum()/total_enrolments:.1f}% of new enrolments are adults, suggesting migration or late adoption.",
        f"3. CHILD ENROLMENT GAP: Only {100*df_enrolment['age_0_5'].sum()/total_enrolments:.1f}% are infants (0-5), a key target for campaigns.",
    ]
    for insight in insights:
        print(f"\n{insight}")

# ============================================================================
# SECTION 12: CREATE SUMMARY INFOGRAPHIC
# ============================================================================


print("\n" + "=" * 80)
print("SECTION 12: SUMMARY INFOGRAPHIC")
print("=" * 80)

# FIX: Calculate zone_summary here to ensure it is defined
# zone_data was created in Section 5.4
zone_summary = zone_data.sum(axis=1).sort_values(ascending=False)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('AADHAAR SOCIETAL TRENDS ANALYSIS - EXECUTIVE SUMMARY', 
             fontsize=20, fontweight='bold', y=0.98)

# Create grid for summary
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# KPI Cards
kpi_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
kpis = [
    ('Total Enrolments', f'{total_enrolments/1e6:.2f}M', '#FF6B6B'),
    ('Biometric Updates', f'{total_biometric/1e6:.2f}M', '#4ECDC4'),
    ('Demographic Updates', f'{total_demographic/1e6:.2f}M', '#45B7D1'),
    ('Total Activities', f'{grand_total/1e6:.2f}M', '#96CEB4')
]

for i, (title, value, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i] if i < 3 else gs[1, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.05, 0.1), 0.9, 0.8, facecolor=color, alpha=0.3, edgecolor=color, linewidth=3))
    ax.text(0.5, 0.65, value, fontsize=28, fontweight='bold', ha='center', va='center', color=color)
    ax.text(0.5, 0.25, title, fontsize=12, ha='center', va='center')
    ax.axis('off')

# Top States Chart
ax_states = fig.add_subplot(gs[1, :])
top_5_combined = state_summary.nlargest(5, 'Total')
x = np.arange(len(top_5_combined))
width = 0.25
ax_states.bar(x - width, top_5_combined['Enrolment'], width, label='Enrolment', color='#FF6B6B')
ax_states.bar(x, top_5_combined['Biometric'], width, label='Biometric', color='#4ECDC4')
ax_states.bar(x + width, top_5_combined['Demographic'], width, label='Demographic', color='#45B7D1')
ax_states.set_xticks(x)
ax_states.set_xticklabels(top_5_combined.index, rotation=15, ha='right')
ax_states.set_title('Top 5 States by Aadhaar Activity', fontsize=14, fontweight='bold')
ax_states.legend()
ax_states.set_ylabel('Count')

# Zone Pie Chart
ax_zone = fig.add_subplot(gs[2, 0])
colors = plt.cm.Set3(np.linspace(0, 1, len(zone_summary)))
ax_zone.pie(zone_summary, labels=zone_summary.index, autopct='%1.0f%%', colors=colors)
ax_zone.set_title('Activity by Zone', fontsize=12, fontweight='bold')

# Age Distribution
ax_age = fig.add_subplot(gs[2, 1])
age_labels = ['0-5 Years', '5-17 Years', '18+ Years']
age_values = [df_enrolment['age_0_5'].sum(), df_enrolment['age_5_17'].sum(), df_enrolment['age_18_greater'].sum()]
ax_age.bar(age_labels, age_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax_age.set_title('Enrolment by Age Group', fontsize=12, fontweight='bold')
ax_age.set_ylabel('Count')

# Key Insights Box
ax_insights = fig.add_subplot(gs[2, 2])
ax_insights.axis('off')
ax_insights.set_xlim(0, 1)
ax_insights.set_ylim(0, 1)

insights_text = """KEY INSIGHTS:

• Demographic updates dominate
  (address/name changes common)

• Adults (18+) lead enrolments
  (late adoption or migration)

• North zone most active
  (large population states)

• Child enrolment gap exists
  (opportunity for early-life campaigns)
"""
ax_insights.text(0.05, 0.95, insights_text, fontsize=10, va='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 family='monospace')
ax_insights.set_title('Key Insights', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'executive_summary.png', dpi=300, bbox_inches='tight')
print("Saved: executive_summary.png")

# ============================================================================
# SECTION 13: SAVE ANALYSIS RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 13: SAVING ANALYSIS RESULTS")
print("=" * 80)

# Save state-level summary
state_summary.to_csv(OUTPUT_PATH / 'state_summary.csv')
print("\nSaved: state_summary.csv")

# Save zone summary
zone_data.to_csv(OUTPUT_PATH / 'zone_summary.csv')
print("Saved: zone_summary.csv")

# Save top districts
top_districts.to_csv(OUTPUT_PATH / 'top_districts.csv', index=False)
print("Saved: top_districts.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll outputs saved to: {OUTPUT_PATH}")