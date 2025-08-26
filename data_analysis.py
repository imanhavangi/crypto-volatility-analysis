import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_data_distribution(training_file='training_data.csv'):
    """
    تحلیل توزیع داده‌ها و بررسی class imbalance
    """
    print("🔍 در حال بارگذاری و تحلیل داده‌ها...")
    
    # بارگذاری داده‌ها
    df = pd.read_csv(training_file)
    
    print(f"📊 تعداد کل نمونه‌ها: {len(df)}")
    print(f"📊 تعداد ستون‌ها: {df.shape[1]}")
    
    # بررسی مقادیر null
    print(f"\n❌ مقادیر null:\n{df.isnull().sum()}")
    
    # تحلیل توزیع کلاس‌های هدف
    print("\n" + "="*50)
    print("🎯 تحلیل توزیع کلاس‌های هدف")
    print("="*50)
    
    # Entry prediction analysis
    entry_counts = df['is_optimal_entry'].value_counts()
    entry_ratio = entry_counts / len(df) * 100
    
    print(f"\n📈 Entry Prediction Distribution:")
    print(f"   False (0): {entry_counts[False]:,} ({entry_ratio[False]:.2f}%)")
    print(f"   True (1):  {entry_counts[True]:,} ({entry_ratio[True]:.2f}%)")
    print(f"   Imbalance Ratio: {entry_counts[False] / entry_counts[True]:.1f}:1")
    
    # Exit prediction analysis
    exit_counts = df['is_optimal_exit'].value_counts()
    exit_ratio = exit_counts / len(df) * 100
    
    print(f"\n📉 Exit Prediction Distribution:")
    print(f"   False (0): {exit_counts[False]:,} ({exit_ratio[False]:.2f}%)")
    print(f"   True (1):  {exit_counts[True]:,} ({exit_ratio[True]:.2f}%)")
    print(f"   Imbalance Ratio: {exit_counts[False] / exit_counts[True]:.1f}:1")
    
    # Future profit analysis
    print(f"\n💰 Future Profit Potential Analysis:")
    profit_stats = df['future_profit_potential'].describe()
    print(profit_stats)
    
    positive_profit = (df['future_profit_potential'] > 0).sum()
    zero_profit = (df['future_profit_potential'] == 0).sum()
    negative_profit = (df['future_profit_potential'] < 0).sum()
    
    print(f"\n   Positive Profit: {positive_profit:,} ({positive_profit/len(df)*100:.2f}%)")
    print(f"   Zero Profit:     {zero_profit:,} ({zero_profit/len(df)*100:.2f}%)")
    print(f"   Negative Profit: {negative_profit:,} ({negative_profit/len(df)*100:.2f}%)")
    
    # ترسیم نمودارها
    plt.figure(figsize=(15, 10))
    
    # Entry prediction distribution
    plt.subplot(2, 3, 1)
    entry_counts.plot(kind='bar', color=['lightcoral', 'lightgreen'])
    plt.title('Entry Prediction Distribution')
    plt.xlabel('is_optimal_entry')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Exit prediction distribution
    plt.subplot(2, 3, 2)
    exit_counts.plot(kind='bar', color=['lightcoral', 'lightblue'])
    plt.title('Exit Prediction Distribution')
    plt.xlabel('is_optimal_exit')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Future profit histogram
    plt.subplot(2, 3, 3)
    plt.hist(df['future_profit_potential'], bins=50, alpha=0.7, color='purple')
    plt.title('Future Profit Potential Distribution')
    plt.xlabel('Future Profit Potential')
    plt.ylabel('Frequency')
    
    # Correlation heatmap of target variables
    plt.subplot(2, 3, 4)
    target_corr = df[['is_optimal_entry', 'is_optimal_exit', 'future_profit_potential']].corr()
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Target Variables Correlation')
    
    # Combined entry/exit analysis
    plt.subplot(2, 3, 5)
    combined_analysis = pd.crosstab(df['is_optimal_entry'], df['is_optimal_exit'], margins=True)
    sns.heatmap(combined_analysis.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues')
    plt.title('Entry vs Exit Cross-tabulation')
    
    # Feature importance analysis (RSI as example)
    plt.subplot(2, 3, 6)
    plt.scatter(df['rsi'], df['future_profit_potential'], alpha=0.1, s=1)
    plt.xlabel('RSI')
    plt.ylabel('Future Profit Potential')
    plt.title('RSI vs Future Profit')
    
    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # محاسبه وزن‌های کلاس برای مدل
    print("\n" + "="*50)
    print("⚖️  پیشنهاد وزن‌های کلاس برای مدل")
    print("="*50)
    
    # Class weights for entry prediction
    total_entry = len(df)
    weight_entry_0 = total_entry / (2 * entry_counts[False])
    weight_entry_1 = total_entry / (2 * entry_counts[True])
    
    print(f"\n📈 Entry Prediction Class Weights:")
    print(f"   Class 0 (False): {weight_entry_0:.4f}")
    print(f"   Class 1 (True):  {weight_entry_1:.4f}")
    
    # Class weights for exit prediction
    weight_exit_0 = total_entry / (2 * exit_counts[False])
    weight_exit_1 = total_entry / (2 * exit_counts[True])
    
    print(f"\n📉 Exit Prediction Class Weights:")
    print(f"   Class 0 (False): {weight_exit_0:.4f}")
    print(f"   Class 1 (True):  {weight_exit_1:.4f}")
    
    return {
        'entry_weights': {0: weight_entry_0, 1: weight_entry_1},
        'exit_weights': {0: weight_exit_0, 1: weight_exit_1},
        'entry_counts': entry_counts,
        'exit_counts': exit_counts,
        'data_shape': df.shape
    }

def suggest_improvements():
    """
    پیشنهادات برای بهبود مدل
    """
    print("\n" + "="*60)
    print("💡 پیشنهادات برای حل مشکل Class Imbalance")
    print("="*60)
    
    suggestions = [
        "1️⃣  استفاده از Class Weighting در مدل",
        "2️⃣  پیاده‌سازی SMOTE برای oversampling",
        "3️⃣  تنظیم threshold برای classification",
        "4️⃣  استفاده از Focal Loss به جای Binary Crossentropy",
        "5️⃣  متریک‌های مناسب: F1-Score, AUC-ROC, Precision-Recall AUC",
        "6️⃣  Stratified sampling در train/test split",
        "7️⃣  Ensemble methods با different thresholds"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
    
    print(f"\n🎯 بهترین راه‌کار: ترکیب Class Weighting + SMOTE + مناسب‌ترین Threshold")

if __name__ == "__main__":
    analysis_results = analyze_data_distribution()
    suggest_improvements() 