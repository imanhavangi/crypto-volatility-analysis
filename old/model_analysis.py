import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.99, gamma=3.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        focal_loss = -alpha_t * tf.math.pow(1 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal_loss)

class ModelAnalyzer:
    """
    تحلیل عمیق کیفیت مدل و پیش‌بینی‌هاش
    """
    
    def __init__(self, model_path='best_improved_model.keras', 
                 model_info_path='model_info.json'):
        print("🔍 شروع تحلیل عمیق مدل...")
        
        # بارگذاری مدل
        self.model = tf.keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
        
        # بارگذاری اطلاعات مدل
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.scaler = StandardScaler()
        self.feature_columns = self.model_info['feature_columns']
        
        print("✅ مدل بارگذاری شد")
    
    def load_and_prepare_data(self, test_size=1000):
        """
        بارگذاری و آماده‌سازی داده‌ها برای تست
        """
        print(f"📥 بارگذاری آخرین {test_size} رکورد برای تست...")
        
        # بارگذاری داده‌ها
        df = pd.read_csv('training_data.csv')
        
        # تبدیل timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # انتخاب آخرین رکوردها برای تست
        df_test = df.tail(test_size).copy()
        
        # آماده‌سازی features
        X = df_test[self.feature_columns].copy()
        y_entry = df_test['is_optimal_entry'].values
        y_exit = df_test['is_optimal_exit'].values
        
        # حذف NaN ها
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y_entry = y_entry[mask]
        y_exit = y_exit[mask]
        
        # Scaling
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ {len(X_scaled)} رکورد آماده شد")
        print(f"📊 Entry: {y_entry.sum()} مورد از {len(y_entry)} ({y_entry.mean()*100:.1f}%)")
        print(f"📊 Exit: {y_exit.sum()} مورد از {len(y_exit)} ({y_exit.mean()*100:.1f}%)")
        
        return X_scaled, y_entry, y_exit, df_test
    
    def analyze_model_predictions(self, X, y_entry, y_exit):
        """
        تحلیل کیفیت پیش‌بینی‌های مدل
        """
        print("\n🎯 تحلیل کیفیت پیش‌بینی‌های مدل...")
        
        # پیش‌بینی مدل
        predictions = self.model.predict(X, verbose=0)
        entry_probs = predictions[0].flatten()
        exit_probs = predictions[1].flatten()
        
        # تحلیل توزیع احتمالات
        print(f"\n📊 توزیع احتمالات Entry:")
        print(f"   میانگین: {entry_probs.mean():.3f}")
        print(f"   انحراف معیار: {entry_probs.std():.3f}")
        print(f"   حداقل: {entry_probs.min():.3f}")
        print(f"   حداکثر: {entry_probs.max():.3f}")
        print(f"   تعداد > 0.5: {(entry_probs > 0.5).sum()} ({(entry_probs > 0.5).mean()*100:.1f}%)")
        print(f"   تعداد > 0.8: {(entry_probs > 0.8).sum()} ({(entry_probs > 0.8).mean()*100:.1f}%)")
        print(f"   تعداد > 0.9: {(entry_probs > 0.9).sum()} ({(entry_probs > 0.9).mean()*100:.1f}%)")
        
        print(f"\n📊 توزیع احتمالات Exit:")
        print(f"   میانگین: {exit_probs.mean():.3f}")
        print(f"   انحراف معیار: {exit_probs.std():.3f}")
        print(f"   حداقل: {exit_probs.min():.3f}")
        print(f"   حداکثر: {exit_probs.max():.3f}")
        print(f"   تعداد > 0.5: {(exit_probs > 0.5).sum()} ({(exit_probs > 0.5).mean()*100:.1f}%)")
        print(f"   تعداد > 0.8: {(exit_probs > 0.8).sum()} ({(exit_probs > 0.8).mean()*100:.1f}%)")
        print(f"   تعداد > 0.9: {(exit_probs > 0.9).sum()} ({(exit_probs > 0.9).mean()*100:.1f}%)")
        
        return entry_probs, exit_probs
    
    def test_different_thresholds(self, entry_probs, exit_probs, y_entry, y_exit):
        """
        تست threshold های مختلف
        """
        print("\n🎚️ تست threshold های مختلف...")
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        
        for threshold in thresholds:
            # Entry predictions
            entry_pred = (entry_probs > threshold).astype(int)
            exit_pred = (exit_probs > threshold).astype(int)
            
            # محاسبه metrics
            entry_accuracy = (entry_pred == y_entry).mean()
            exit_accuracy = (exit_pred == y_exit).mean()
            
            # True/False Positives
            entry_tp = ((entry_pred == 1) & (y_entry == 1)).sum()
            entry_fp = ((entry_pred == 1) & (y_entry == 0)).sum()
            entry_tn = ((entry_pred == 0) & (y_entry == 0)).sum()
            entry_fn = ((entry_pred == 0) & (y_entry == 1)).sum()
            
            exit_tp = ((exit_pred == 1) & (y_exit == 1)).sum()
            exit_fp = ((exit_pred == 1) & (y_exit == 0)).sum()
            exit_tn = ((exit_pred == 0) & (y_exit == 0)).sum()
            exit_fn = ((exit_pred == 0) & (y_exit == 1)).sum()
            
            # Precision & Recall
            entry_precision = entry_tp / (entry_tp + entry_fp) if (entry_tp + entry_fp) > 0 else 0
            entry_recall = entry_tp / (entry_tp + entry_fn) if (entry_tp + entry_fn) > 0 else 0
            entry_f1 = 2 * entry_precision * entry_recall / (entry_precision + entry_recall) if (entry_precision + entry_recall) > 0 else 0
            
            exit_precision = exit_tp / (exit_tp + exit_fp) if (exit_tp + exit_fp) > 0 else 0
            exit_recall = exit_tp / (exit_tp + exit_fn) if (exit_tp + exit_fn) > 0 else 0
            exit_f1 = 2 * exit_precision * exit_recall / (exit_precision + exit_recall) if (exit_precision + exit_recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'entry_signals': entry_pred.sum(),
                'exit_signals': exit_pred.sum(),
                'entry_accuracy': entry_accuracy,
                'exit_accuracy': exit_accuracy,
                'entry_precision': entry_precision,
                'entry_recall': entry_recall,
                'entry_f1': entry_f1,
                'exit_precision': exit_precision,
                'exit_recall': exit_recall,
                'exit_f1': exit_f1
            })
        
        # نمایش نتایج
        results_df = pd.DataFrame(results)
        print("\n📋 نتایج threshold های مختلف:")
        print("="*100)
        print(f"{'Threshold':<10} {'Entry Sigs':<12} {'Exit Sigs':<11} {'Entry Acc':<11} {'Exit Acc':<10} {'Entry F1':<10} {'Exit F1':<10}")
        print("="*100)
        
        for _, row in results_df.iterrows():
            print(f"{row['threshold']:<10.1f} {row['entry_signals']:<12.0f} {row['exit_signals']:<11.0f} {row['entry_accuracy']:<11.3f} {row['exit_accuracy']:<10.3f} {row['entry_f1']:<10.3f} {row['exit_f1']:<10.3f}")
        
        return results_df
    
    def analyze_prediction_quality(self, entry_probs, exit_probs, y_entry, y_exit, df_test):
        """
        تحلیل کیفیت پیش‌بینی در نقاط مختلف
        """
        print("\n🔬 تحلیل کیفیت پیش‌بینی در نقاط مختلف...")
        
        # ایجاد DataFrame برای تحلیل
        analysis_df = df_test.tail(len(entry_probs)).copy()
        analysis_df['entry_prob'] = entry_probs
        analysis_df['exit_prob'] = exit_probs
        analysis_df['actual_entry'] = y_entry
        analysis_df['actual_exit'] = y_exit
        
        # تحلیل نقاط با احتمال بالا
        high_confidence_entry = analysis_df[analysis_df['entry_prob'] > 0.8]
        high_confidence_exit = analysis_df[analysis_df['exit_prob'] > 0.8]
        
        print(f"\n🎯 نقاط با اعتماد بالا (>0.8):")
        print(f"   Entry: {len(high_confidence_entry)} نقطه")
        if len(high_confidence_entry) > 0:
            accuracy_entry = (high_confidence_entry['actual_entry'] == 1).mean()
            print(f"   دقت Entry: {accuracy_entry:.3f} ({accuracy_entry*100:.1f}%)")
        
        print(f"   Exit: {len(high_confidence_exit)} نقطه")
        if len(high_confidence_exit) > 0:
            accuracy_exit = (high_confidence_exit['actual_exit'] == 1).mean()
            print(f"   دقت Exit: {accuracy_exit:.3f} ({accuracy_exit*100:.1f}%)")
        
        # تحلیل correlation بین price movement و predictions
        analysis_df['price_change_future'] = analysis_df['close'].shift(-5) / analysis_df['close'] - 1
        
        if 'price_change_future' in analysis_df.columns:
            correlation_entry = np.corrcoef(analysis_df['entry_prob'].iloc[:-5], 
                                          analysis_df['price_change_future'].iloc[:-5])[0,1]
            correlation_exit = np.corrcoef(analysis_df['exit_prob'].iloc[:-5], 
                                         -analysis_df['price_change_future'].iloc[:-5])[0,1]
            
            print(f"\n📈 Correlation با حرکت قیمت:")
            print(f"   Entry vs Price Rise: {correlation_entry:.3f}")
            print(f"   Exit vs Price Fall: {correlation_exit:.3f}")
        
        return analysis_df
    
    def recommend_optimal_strategy(self, results_df):
        """
        توصیه استراتژی بهینه
        """
        print("\n💡 توصیه استراتژی بهینه:")
        print("="*50)
        
        # یافتن بهترین threshold برای تعداد سیگنال متعادل
        moderate_signals = results_df[(results_df['entry_signals'] >= 50) & 
                                    (results_df['entry_signals'] <= 200)]
        
        if len(moderate_signals) > 0:
            best_moderate = moderate_signals.loc[moderate_signals['entry_f1'].idxmax()]
            print(f"🎯 برای تعداد متعادل سیگنال (50-200):")
            print(f"   Threshold بهینه: {best_moderate['threshold']}")
            print(f"   تعداد سیگنال Entry: {best_moderate['entry_signals']:.0f}")
            print(f"   Entry F1: {best_moderate['entry_f1']:.3f}")
            print(f"   Entry Precision: {best_moderate['entry_precision']:.3f}")
            print(f"   Entry Recall: {best_moderate['entry_recall']:.3f}")
        
        # یافتن بهترین threshold برای کیفیت
        high_quality = results_df[results_df['entry_f1'] > 0]
        if len(high_quality) > 0:
            best_quality = high_quality.loc[high_quality['entry_f1'].idxmax()]
            print(f"\n🏆 برای بالاترین کیفیت:")
            print(f"   Threshold بهینه: {best_quality['threshold']}")
            print(f"   تعداد سیگنال Entry: {best_quality['entry_signals']:.0f}")
            print(f"   Entry F1: {best_quality['entry_f1']:.3f}")
            print(f"   Entry Precision: {best_quality['entry_precision']:.3f}")
            print(f"   Entry Recall: {best_quality['entry_recall']:.3f}")
        
        # یافتن threshold برای تعداد سیگنال زیاد
        high_signals = results_df[results_df['entry_signals'] >= 100]
        if len(high_signals) > 0:
            best_high_volume = high_signals.loc[high_signals['entry_f1'].idxmax()]
            print(f"\n🔄 برای تعداد زیاد سیگنال (100+):")
            print(f"   Threshold بهینه: {best_high_volume['threshold']}")
            print(f"   تعداد سیگنال Entry: {best_high_volume['entry_signals']:.0f}")
            print(f"   Entry F1: {best_high_volume['entry_f1']:.3f}")
            print(f"   Entry Precision: {best_high_volume['entry_precision']:.3f}")
            print(f"   Entry Recall: {best_high_volume['entry_recall']:.3f}")
    
    def run_complete_analysis(self):
        """
        اجرای تحلیل کامل
        """
        print("🚀 شروع تحلیل کامل مدل و استراتژی")
        print("="*60)
        
        # بارگذاری داده‌ها
        X, y_entry, y_exit, df_test = self.load_and_prepare_data(2000)
        
        # تحلیل پیش‌بینی‌ها
        entry_probs, exit_probs = self.analyze_model_predictions(X, y_entry, y_exit)
        
        # تست threshold های مختلف
        results_df = self.test_different_thresholds(entry_probs, exit_probs, y_entry, y_exit)
        
        # تحلیل کیفیت
        analysis_df = self.analyze_prediction_quality(entry_probs, exit_probs, y_entry, y_exit, df_test)
        
        # توصیه‌ها
        self.recommend_optimal_strategy(results_df)
        
        print("\n" + "="*60)
        print("📊 خلاصه یافته‌ها:")
        print("="*60)
        
        # نتیجه‌گیری نهایی
        avg_entry_prob = entry_probs.mean()
        avg_exit_prob = exit_probs.mean()
        
        if avg_entry_prob < 0.3 and avg_exit_prob < 0.3:
            print("❌ مدل اعتماد پایینی به پیش‌بینی‌هاش دارد")
            print("💡 توصیه: مدل نیاز به بازآموزی دارد")
        elif avg_entry_prob > 0.7 or avg_exit_prob > 0.7:
            print("⚠️ مدل ممکن است overfit باشد")
            print("💡 توصیه: بررسی validation set")
        else:
            print("✅ توزیع احتمالات منطقی است")
            print("💡 توصیه: تنظیم threshold ها کافیست")
        
        # بررسی تعداد سیگنال‌ها
        signals_05 = (entry_probs > 0.5).sum()
        if signals_05 < 50:
            print(f"⚠️ تعداد سیگنال کم ({signals_05}) - threshold کاهش دهید")
        elif signals_05 > 500:
            print(f"⚠️ تعداد سیگنال زیاد ({signals_05}) - threshold افزایش دهید")
        else:
            print(f"✅ تعداد سیگنال مناسب ({signals_05})")
        
        return results_df, analysis_df

if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    results_df, analysis_df = analyzer.run_complete_analysis()



