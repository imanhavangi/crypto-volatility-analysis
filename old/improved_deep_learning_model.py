import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, f1_score,
                            precision_score, recall_score, fbeta_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# تنظیم GPU اگر موجود باشد
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss بهبود یافته برای حل مشکل class imbalance شدید
    """
    def __init__(self, alpha=0.99, gamma=3.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # افزایش alpha برای کلاس اقلیت
        self.gamma = gamma  # افزایش gamma برای تمرکز بیشتر بر نمونه‌های سخت

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        focal_loss = -alpha_t * tf.math.pow(1 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal_loss)

class ImbalancedDataProcessor:
    """
    کلاس تخصصی برای کار با داده‌های نامتعادل
    """
    def __init__(self):
        self.positive_ratio = None
        self.negative_ratio = None
        
    def analyze_imbalance(self, y):
        """تحلیل میزان نابرابری"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        self.positive_ratio = counts[1] / total if len(counts) > 1 else 0
        self.negative_ratio = counts[0] / total if len(counts) > 0 else 1
        
        print(f"📊 Class Distribution Analysis:")
        print(f"   - Negative (0): {counts[0]:,} ({self.negative_ratio:.1%})")
        print(f"   - Positive (1): {counts[1]:,} ({self.positive_ratio:.1%})" if len(counts) > 1 else "   - Positive (1): 0 (0%)")
        print(f"   - Imbalance Ratio: 1:{counts[0]/counts[1]:.0f}" if len(counts) > 1 and counts[1] > 0 else "   - Severe Imbalance!")
        
        return self.positive_ratio, self.negative_ratio
    
    def create_balanced_dataset(self, X, y, method='hybrid'):
        """ایجاد dataset متعادل با روش‌های مختلف"""
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import ADASYN
        from imblearn.combine import SMOTEENN
        
        print(f"🔄 Creating balanced dataset using {method} method...")
        
        if method == 'smote_enn':
            # ترکیب SMOTE و Edited Nearest Neighbours
            sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=3))
        elif method == 'adasyn':
            # ADASYN برای تولید نمونه‌های سینتتیک هوشمندتر
            sampler = ADASYN(random_state=42, n_neighbors=3)
        elif method == 'hybrid':
            # ترکیب undersampling + oversampling
            # ابتدا کلاس اکثریت را کم می‌کنیم
            under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            X_under, y_under = under_sampler.fit_resample(X, y)
            # سپس کلاس اقلیت را افزایش می‌دهیم
            over_sampler = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = over_sampler.fit_resample(X_under, y_under)
            return X_balanced, y_balanced
        else:
            # فقط SMOTE
            sampler = SMOTE(random_state=42, k_neighbors=3)
            
        try:
            X_balanced, y_balanced = sampler.fit_resample(X, y)
            print(f"✅ Balanced dataset created. New shape: {X_balanced.shape}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"⚠️ Balancing failed: {e}")
            print("🔄 Falling back to basic SMOTE...")
            basic_smote = SMOTE(random_state=42, k_neighbors=1)
            return basic_smote.fit_resample(X, y)

class ImprovedTradingModel:
    def __init__(self, input_features=20, use_focal_loss=True, balancing_method='hybrid'):
        """
        مدل بهبود یافته برای پیش‌بینی ترید با حل مشکل class imbalance شدید
        
        Args:
            input_features: تعداد فیچرهای ورودی
            use_focal_loss: استفاده از Focal Loss
            balancing_method: روش متعادل کردن داده‌ها ('hybrid', 'smote_enn', 'adasyn', 'smote')
        """
        self.input_features = input_features
        self.use_focal_loss = use_focal_loss
        self.balancing_method = balancing_method
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.class_weights_entry = None
        self.class_weights_exit = None
        self.data_processor = ImbalancedDataProcessor()
        self.threshold_entry = 0.5
        self.threshold_exit = 0.5
        self.training_history = None
        
    def calculate_class_weights(self, y_entry, y_exit):
        """
        محاسبه وزن‌های کلاس بر اساس inverse frequency
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Entry weights
        entry_classes = np.unique(y_entry)
        entry_weights = compute_class_weight('balanced', classes=entry_classes, y=y_entry)
        self.class_weights_entry = dict(zip(entry_classes, entry_weights))
        
        # Exit weights  
        exit_classes = np.unique(y_exit)
        exit_weights = compute_class_weight('balanced', classes=exit_classes, y=y_exit)
        self.class_weights_exit = dict(zip(exit_classes, exit_weights))
        
        print(f"🏆 Entry Class Weights: {self.class_weights_entry}")
        print(f"🏆 Exit Class Weights: {self.class_weights_exit}")
        
    def create_model_architecture(self):
        """
        ایجاد معماری مدل بهبود یافته با تمرکز بر class imbalance
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_features,), name='features')
        
        # Feature engineering layers با regularization بیشتر
        x = layers.Dense(512, activation='swish', name='feature_extract_1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='swish', name='feature_extract_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='swish', name='feature_extract_3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Enhanced attention mechanism
        attention = layers.Dense(128, activation='tanh', name='attention_weights')(x)
        attention = layers.Dense(64, activation='tanh', name='attention_intermediate')(attention)
        attention = layers.Dense(1, activation='sigmoid', name='attention_scores')(attention)
        x_attended = layers.Multiply(name='attended_features')([x, attention])
        
        # Residual connection
        x_residual = layers.Dense(128, activation='swish')(inputs)
        x_combined = layers.Add()([x_attended, x_residual])
        
        # Shared representation with more capacity
        shared = layers.Dense(96, activation='swish', name='shared_representation')(x_combined)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.15)(shared)
        
        # Entry prediction branch with specialized architecture for imbalanced data
        entry_branch = layers.Dense(64, activation='swish', name='entry_branch_1')(shared)
        entry_branch = layers.BatchNormalization()(entry_branch)
        entry_branch = layers.Dropout(0.1)(entry_branch)
        entry_branch = layers.Dense(32, activation='swish', name='entry_branch_2')(entry_branch)
        entry_branch = layers.Dropout(0.05)(entry_branch)
        # استفاده از bias initializer مناسب برای imbalanced data
        initial_bias = np.log([self.data_processor.positive_ratio / self.data_processor.negative_ratio]) if self.data_processor.positive_ratio else -2.0
        entry_output = layers.Dense(1, activation='sigmoid', 
                                   bias_initializer=keras.initializers.Constant(initial_bias),
                                   name='entry_prediction')(entry_branch)
        
        # Exit prediction branch
        exit_branch = layers.Dense(64, activation='swish', name='exit_branch_1')(shared)
        exit_branch = layers.BatchNormalization()(exit_branch)
        exit_branch = layers.Dropout(0.1)(exit_branch)
        exit_branch = layers.Dense(32, activation='swish', name='exit_branch_2')(exit_branch)
        exit_branch = layers.Dropout(0.05)(exit_branch)
        exit_output = layers.Dense(1, activation='sigmoid',
                                  bias_initializer=keras.initializers.Constant(initial_bias),
                                  name='exit_prediction')(exit_branch)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=[entry_output, exit_output])
        
        return model
    
    def compile_model(self):
        """
        کامپایل مدل با loss functions و metrics مناسب برای imbalanced data
        """
        if self.use_focal_loss:
            # پارامترهای بهینه‌شده برای class imbalance شدید
            entry_loss = FocalLoss(alpha=0.99, gamma=3.0)  
            exit_loss = FocalLoss(alpha=0.99, gamma=3.0)
        else:
            entry_loss = 'binary_crossentropy'
            exit_loss = 'binary_crossentropy'
            
        # استفاده از AdamW برای regularization بهتر
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
            
        self.model.compile(
            optimizer=optimizer,
            loss={
                'entry_prediction': entry_loss,
                'exit_prediction': exit_loss
            },
            loss_weights={
                'entry_prediction': 1.0,
                'exit_prediction': 1.0
            },
            metrics={
                'entry_prediction': [
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalseNegatives(name='fn')
                ],
                'exit_prediction': [
                    'accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalseNegatives(name='fn')
                ]
            }
        )
        
    def prepare_data(self, df):
        """
        آماده‌سازی داده‌ها با feature engineering و تکنیک‌های پیشرفته برای imbalanced data
        """
        print("🔧 آماده‌سازی داده‌ها با تکنیک‌های پیشرفته...")
        
        # انتخاب فیچرها
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'is_optimal_entry', 'is_optimal_exit', 'future_profit_potential']]
        
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y_entry = df['is_optimal_entry'].values.astype(int)
        y_exit = df['is_optimal_exit'].values.astype(int)
        
        print(f"📊 تعداد فیچرها: {len(feature_cols)}")
        print(f"📊 شکل داده‌ها: {X.shape}")
        
        # تحلیل class imbalance
        print("\n🔍 تحلیل Entry Class Distribution:")
        self.data_processor.analyze_imbalance(y_entry)
        print("\n🔍 تحلیل Exit Class Distribution:")
        self.data_processor.analyze_imbalance(y_exit)
        
        # ایجاد target ترکیبی برای stratified split
        y_combined = y_entry * 2 + y_exit  # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        print(f"\n📊 توزیع ترکیبی: {np.bincount(y_combined)}")
        
        # تقسیم داده‌ها با Stratified Split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y_combined))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_entry_train, y_entry_test = y_entry[train_idx], y_entry[test_idx]
        y_exit_train, y_exit_test = y_exit[train_idx], y_exit[test_idx]
        
        # Feature scaling
        print("📐 Feature Scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # اعمال تکنیک‌های متعادل کردن داده‌ها
        print(f"\n🎯 اعمال روش {self.balancing_method} برای Entry...")
        try:
            X_entry_balanced, y_entry_balanced = self.data_processor.create_balanced_dataset(
                X_train_scaled, y_entry_train, method=self.balancing_method
            )
            print(f"✅ Entry balancing successful. New shape: {X_entry_balanced.shape}")
            print(f"📊 Entry distribution after balancing: {np.bincount(y_entry_balanced)}")
        except Exception as e:
            print(f"⚠️ Entry balancing failed: {e}")
            X_entry_balanced, y_entry_balanced = X_train_scaled, y_entry_train
        
        print(f"\n🎯 اعمال روش {self.balancing_method} برای Exit...")
        try:
            X_exit_balanced, y_exit_balanced = self.data_processor.create_balanced_dataset(
                X_train_scaled, y_exit_train, method=self.balancing_method
            )
            print(f"✅ Exit balancing successful. New shape: {X_exit_balanced.shape}")
            print(f"📊 Exit distribution after balancing: {np.bincount(y_exit_balanced)}")
        except Exception as e:
            print(f"⚠️ Exit balancing failed: {e}")
            X_exit_balanced, y_exit_balanced = X_train_scaled, y_exit_train
        
        # استفاده از داده‌های متعادل‌شده برای Entry (معمولاً balanced dataset بزرگتر است)
        if len(X_entry_balanced) >= len(X_exit_balanced):
            X_train_final = X_entry_balanced
            y_entry_final = y_entry_balanced
            # برای Exit، نمونه‌برداری مجدد برای match کردن با Entry
            if len(X_exit_balanced) < len(X_entry_balanced):
                indices = np.random.choice(len(X_exit_balanced), size=len(X_entry_balanced), replace=True)
                X_exit_resampled = X_exit_balanced[indices]
                y_exit_final = y_exit_balanced[indices]
            else:
                y_exit_final = y_exit_balanced[:len(X_entry_balanced)]
        else:
            X_train_final = X_exit_balanced
            y_exit_final = y_exit_balanced
            # برای Entry، نمونه‌برداری مجدد
            indices = np.random.choice(len(X_entry_balanced), size=len(X_exit_balanced), replace=True)
            X_entry_resampled = X_entry_balanced[indices]
            y_entry_final = y_entry_balanced[indices]
        
        # محاسبه class weights
        self.calculate_class_weights(y_entry_final, y_exit_final)
        
        print(f"\n✅ نهایی‌سازی داده‌ها:")
        print(f"   - Training samples: {len(X_train_final):,}")
        print(f"   - Test samples: {len(X_test_scaled):,}")
        print(f"   - Entry positive ratio: {np.mean(y_entry_final):.3f}")
        print(f"   - Exit positive ratio: {np.mean(y_exit_final):.3f}")
        
        return {
            'X_train': X_train_final,
            'X_test': X_test_scaled,
            'y_entry_train': y_entry_final,
            'y_entry_test': y_entry_test,
            'y_exit_train': y_exit_final,
            'y_exit_test': y_exit_test
        }
    
    def find_optimal_threshold(self, X_val, y_entry_val, y_exit_val):
        """
        یافتن بهترین threshold با استفاده از چندین metric برای imbalanced data
        """
        print("🎯 جستجوی بهترین threshold با روش‌های پیشرفته...")
        
        # پیش‌بینی احتمالات
        entry_probs, exit_probs = self.model.predict(X_val, verbose=0)
        entry_probs = entry_probs.flatten()
        exit_probs = exit_probs.flatten()
        
        # محدوده threshold گسترده‌تر
        thresholds = np.arange(0.05, 0.95, 0.025)
        
        def evaluate_threshold(y_true, y_probs, thresholds):
            """محاسبه metrics مختلف برای هر threshold"""
            results = []
            for threshold in thresholds:
                y_pred = (y_probs > threshold).astype(int)
                
                # محاسبه metrics مختلف
                try:
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    precision = precision_score(y_true, y_pred, zero_division=0) 
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    # F-beta score که recall را بیشتر وزن دهد (مهم برای imbalanced data)
                    fbeta = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                    # Matthews Correlation Coefficient - مناسب برای imbalanced data
                    mcc = matthews_corrcoef(y_true, y_pred)
                    
                    # ترکیب weighted از metrics
                    combined_score = (0.3 * f1) + (0.4 * fbeta) + (0.3 * mcc)
                    
                    results.append({
                        'threshold': threshold,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'fbeta': fbeta,
                        'mcc': mcc,
                        'combined': combined_score
                    })
                except:
                    results.append({
                        'threshold': threshold,
                        'f1': 0, 'precision': 0, 'recall': 0,
                        'fbeta': 0, 'mcc': 0, 'combined': 0
                    })
            
            return results
        
        # Entry threshold optimization
        print("🔍 بهینه‌سازی Entry threshold...")
        entry_results = evaluate_threshold(y_entry_val, entry_probs, thresholds)
        best_entry = max(entry_results, key=lambda x: x['combined'])
        
        # Exit threshold optimization
        print("🔍 بهینه‌سازی Exit threshold...")
        exit_results = evaluate_threshold(y_exit_val, exit_probs, thresholds)
        best_exit = max(exit_results, key=lambda x: x['combined'])
        
        self.threshold_entry = best_entry['threshold']
        self.threshold_exit = best_exit['threshold']
        
        print(f"\n✅ بهترین Entry Threshold: {self.threshold_entry:.3f}")
        print(f"   📊 F1: {best_entry['f1']:.3f}, F-beta: {best_entry['fbeta']:.3f}, MCC: {best_entry['mcc']:.3f}")
        print(f"   📊 Precision: {best_entry['precision']:.3f}, Recall: {best_entry['recall']:.3f}")
        
        print(f"\n✅ بهترین Exit Threshold: {self.threshold_exit:.3f}")
        print(f"   📊 F1: {best_exit['f1']:.3f}, F-beta: {best_exit['fbeta']:.3f}, MCC: {best_exit['mcc']:.3f}")
        print(f"   📊 Precision: {best_exit['precision']:.3f}, Recall: {best_exit['recall']:.3f}")
        
        return {
            'entry_results': entry_results,
            'exit_results': exit_results,
            'best_entry': best_entry,
            'best_exit': best_exit
        }
    
    def train(self, training_file='training_data.csv', epochs=50, batch_size=256):
        """
        آموزش مدل با تمام بهینه‌سازی‌ها
        """
        print("🚀 شروع آموزش مدل بهبود یافته...")
        
        # بارگذاری داده‌ها
        df = pd.read_csv(training_file)
        
        # آماده‌سازی داده‌ها
        data = self.prepare_data(df)
        
        # تنظیم تعداد فیچرها
        self.input_features = len(self.feature_columns)
        
        # ایجاد مدل
        self.model = self.create_model_architecture()
        self.compile_model()
        
        print("\n📋 خلاصه معماری مدل:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_improved_model.keras', save_best_only=True, monitor='val_loss', verbose=1
            )
        ]
        
        # ایجاد validation split
        val_split = 0.15
        val_size = int(len(data['X_train']) * val_split)
        indices = np.random.permutation(len(data['X_train']))
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train_final = data['X_train'][train_indices]
        X_val = data['X_train'][val_indices]
        y_entry_train_final = data['y_entry_train'][train_indices]
        y_entry_val = data['y_entry_train'][val_indices]
        y_exit_train_final = data['y_exit_train'][train_indices]
        y_exit_val = data['y_exit_train'][val_indices]
        
        # آموزش مدل
        print("\n🎓 شروع آموزش...")
        history = self.model.fit(
            X_train_final,
            {
                'entry_prediction': y_entry_train_final,
                'exit_prediction': y_exit_train_final
            },
            validation_data=(
                X_val,
                {
                    'entry_prediction': y_entry_val,
                    'exit_prediction': y_exit_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # یافتن بهترین threshold
        self.find_optimal_threshold(X_val, y_entry_val, y_exit_val)
        
        # ارزیابی نهایی
        self.evaluate_model(data['X_test'], data['y_entry_test'], data['y_exit_test'])
        
        # رسم نمودارهای آموزش
        self.plot_training_history(history)
        
        return history
    
    def evaluate_model(self, X_test, y_entry_test, y_exit_test):
        """
        ارزیابی دقیق مدل با metrics مناسب
        """
        print("\n📈 ارزیابی نهایی مدل...")
        
        # پیش‌بینی احتمالات
        entry_probs, exit_probs = self.model.predict(X_test, verbose=0)
        
        # پیش‌بینی با threshold بهینه
        entry_pred = (entry_probs > self.threshold_entry).astype(int).flatten()
        exit_pred = (exit_probs > self.threshold_exit).astype(int).flatten()
        
        print("\n🎯 نتایج Entry Prediction:")
        print(classification_report(y_entry_test, entry_pred, target_names=['No Entry', 'Entry']))
        
        print("\n🎯 نتایج Exit Prediction:")
        print(classification_report(y_exit_test, exit_pred, target_names=['No Exit', 'Exit']))
        
        # AUC scores
        try:
            entry_auc = roc_auc_score(y_entry_test, entry_probs)
            exit_auc = roc_auc_score(y_exit_test, exit_probs)
            print(f"\n🏆 Entry AUC-ROC: {entry_auc:.4f}")
            print(f"🏆 Exit AUC-ROC: {exit_auc:.4f}")
        except:
            print("⚠️ نمی‌توان AUC محاسبه کرد (احتمالاً فقط یک کلاس در test set)")
        
        # Confusion matrices
        self.plot_confusion_matrices(y_entry_test, entry_pred, y_exit_test, exit_pred)
        
    def plot_confusion_matrices(self, y_entry_true, y_entry_pred, y_exit_true, y_exit_pred):
        """
        رسم confusion matrices
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        cm_entry = confusion_matrix(y_entry_true, y_entry_pred)
        sns.heatmap(cm_entry, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Entry', 'Entry'],
                   yticklabels=['No Entry', 'Entry'])
        plt.title('Entry Prediction Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        cm_exit = confusion_matrix(y_exit_true, y_exit_pred)
        sns.heatmap(cm_exit, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['No Exit', 'Exit'],
                   yticklabels=['No Exit', 'Exit'])
        plt.title('Exit Prediction Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('improved_model_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history):
        """
        رسم نمودارهای آموزش
        """
        plt.figure(figsize=(15, 10))
        
        # Loss plots
        plt.subplot(2, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Entry metrics
        plt.subplot(2, 3, 2)
        plt.plot(history.history['entry_prediction_precision'], label='Train Precision')
        plt.plot(history.history['val_entry_prediction_precision'], label='Val Precision')
        plt.plot(history.history['entry_prediction_recall'], label='Train Recall')
        plt.plot(history.history['val_entry_prediction_recall'], label='Val Recall')
        plt.title('Entry Prediction Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # Exit metrics
        plt.subplot(2, 3, 3)
        plt.plot(history.history['exit_prediction_precision'], label='Train Precision')
        plt.plot(history.history['val_exit_prediction_precision'], label='Val Precision')
        plt.plot(history.history['exit_prediction_recall'], label='Train Recall')
        plt.plot(history.history['val_exit_prediction_recall'], label='Val Recall')
        plt.title('Exit Prediction Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # AUC plots
        plt.subplot(2, 3, 4)
        plt.plot(history.history['entry_prediction_auc'], label='Entry AUC')
        plt.plot(history.history['val_entry_prediction_auc'], label='Val Entry AUC')
        plt.plot(history.history['exit_prediction_auc'], label='Exit AUC')
        plt.plot(history.history['val_exit_prediction_auc'], label='Val Exit AUC')
        plt.title('AUC Scores')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        # PR AUC plots
        plt.subplot(2, 3, 5)
        plt.plot(history.history['entry_prediction_pr_auc'], label='Entry PR-AUC')
        plt.plot(history.history['val_entry_prediction_pr_auc'], label='Val Entry PR-AUC')
        plt.plot(history.history['exit_prediction_pr_auc'], label='Exit PR-AUC')
        plt.plot(history.history['val_exit_prediction_pr_auc'], label='Val Exit PR-AUC')
        plt.title('Precision-Recall AUC')
        plt.xlabel('Epoch')
        plt.ylabel('PR-AUC')
        plt.legend()
        
        # Learning rate
        plt.subplot(2, 3, 6)
        if 'lr' in history.history:
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('improved_model_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # ایجاد و آموزش مدل بهبود یافته با تکنیک‌های پیشرفته
    model = ImprovedTradingModel(
        use_focal_loss=True,
        balancing_method='hybrid'  # hybrid، smote_enn، adasyn، smote
    )
    
    print("🚀 شروع آموزش مدل بهبود یافته با حل مشکل Class Imbalance شدید")
    print("🔧 تکنیک‌های اعمال شده:")
    print("   - Focal Loss با پارامترهای بهینه (alpha=0.99, gamma=3.0)")
    print("   - Hybrid Balancing (UnderSampling + SMOTE)")
    print("   - Architecture بهبود یافته با Residual Connection")
    print("   - AdamW Optimizer با Weight Decay")
    print("   - Advanced Threshold Optimization")
    print("="*70)
    
    history = model.train(
        training_file='training_data.csv',
        epochs=20,  # افزایش epochs برای convergence بهتر
        batch_size=256  # کاهش batch size برای stability بیشتر
    )
    
    print("\n🎉 آموزش کامل شد!")
    print(f"🎯 Entry Threshold: {model.threshold_entry:.3f}")
    print(f"🎯 Exit Threshold: {model.threshold_exit:.3f}")
    print(f"💾 مدل ذخیره شده در: best_improved_model.keras")
    
    # ذخیره اطلاعات مدل
    model_info = {
        'entry_threshold': model.threshold_entry,
        'exit_threshold': model.threshold_exit,
        'feature_columns': model.feature_columns,
        'balancing_method': model.balancing_method,
        'training_completed': datetime.now().isoformat()
    }
    
    import json
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"📋 اطلاعات مدل ذخیره شده در: model_info.json") 