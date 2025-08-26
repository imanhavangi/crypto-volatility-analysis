import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
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
    Focal Loss برای حل مشکل class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
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

class ImprovedTradingModel:
    def __init__(self, input_features=20, use_focal_loss=True):
        """
        مدل بهبود یافته برای پیش‌بینی ترید با حل مشکل class imbalance
        
        Args:
            input_features: تعداد فیچرهای ورودی
            use_focal_loss: استفاده از Focal Loss
        """
        self.input_features = input_features
        self.use_focal_loss = use_focal_loss
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.class_weights_entry = None
        self.class_weights_exit = None
        self.smote = SMOTE(random_state=42, k_neighbors=3)
        self.threshold_entry = 0.5
        self.threshold_exit = 0.5
        
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
        ایجاد معماری مدل با attention mechanism
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_features,), name='features')
        
        # Feature engineering layers
        x = layers.Dense(256, activation='relu', name='feature_extract_1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu', name='feature_extract_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Attention mechanism
        attention = layers.Dense(128, activation='tanh', name='attention_weights')(x)
        attention = layers.Dense(1, activation='sigmoid', name='attention_scores')(attention)
        x_attended = layers.Multiply(name='attended_features')([x, attention])
        
        # Shared representation
        shared = layers.Dense(64, activation='relu', name='shared_representation')(x_attended)
        shared = layers.Dropout(0.1)(shared)
        
        # Entry prediction branch
        entry_branch = layers.Dense(32, activation='relu', name='entry_branch')(shared)
        entry_branch = layers.Dropout(0.1)(entry_branch)
        entry_output = layers.Dense(1, activation='sigmoid', name='entry_prediction')(entry_branch)
        
        # Exit prediction branch
        exit_branch = layers.Dense(32, activation='relu', name='exit_branch')(shared)
        exit_branch = layers.Dropout(0.1)(exit_branch)
        exit_output = layers.Dense(1, activation='sigmoid', name='exit_prediction')(exit_branch)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=[entry_output, exit_output])
        
        return model
    
    def compile_model(self):
        """
        کامپایل مدل با loss functions و metrics مناسب
        """
        if self.use_focal_loss:
            entry_loss = FocalLoss(alpha=0.75, gamma=2.0)  # alpha بالاتر برای کلاس اقلیت
            exit_loss = FocalLoss(alpha=0.75, gamma=2.0)
        else:
            entry_loss = 'binary_crossentropy'
            exit_loss = 'binary_crossentropy'
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc')
                ],
                'exit_prediction': [
                    'accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc')
                ]
            }
        )
        
    def prepare_data(self, df):
        """
        آماده‌سازی داده‌ها با feature engineering و SMOTE
        """
        print("🔧 آماده‌سازی داده‌ها...")
        
        # انتخاب فیچرها
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'is_optimal_entry', 'is_optimal_exit', 'future_profit_potential']]
        
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y_entry = df['is_optimal_entry'].values.astype(int)
        y_exit = df['is_optimal_exit'].values.astype(int)
        
        print(f"📊 تعداد فیچرها: {len(feature_cols)}")
        print(f"📊 شکل داده‌ها: {X.shape}")
        
        # ایجاد target ترکیبی برای SMOTE
        y_combined = y_entry * 2 + y_exit  # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        print(f"📊 توزیع ترکیبی قبل از SMOTE: {np.bincount(y_combined)}")
        
        # تقسیم داده‌ها با Stratified Split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y_combined))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_entry_train, y_entry_test = y_entry[train_idx], y_entry[test_idx]
        y_exit_train, y_exit_test = y_exit[train_idx], y_exit[test_idx]
        y_combined_train = y_combined[train_idx]
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SMOTE application
        print("🎯 اعمال SMOTE...")
        try:
            X_train_balanced, y_combined_balanced = self.smote.fit_resample(X_train_scaled, y_combined_train)
            
            # تجزیه مجدد y_combined
            y_entry_balanced = (y_combined_balanced >= 2).astype(int)
            y_exit_balanced = (y_combined_balanced % 2).astype(int)
            
            print(f"📊 توزیع Entry بعد از SMOTE: {np.bincount(y_entry_balanced)}")
            print(f"📊 توزیع Exit بعد از SMOTE: {np.bincount(y_exit_balanced)}")
            
        except Exception as e:
            print(f"⚠️  SMOTE ناموفق: {e}")
            print("🔄 ادامه بدون SMOTE...")
            X_train_balanced = X_train_scaled
            y_entry_balanced = y_entry_train
            y_exit_balanced = y_exit_train
        
        # محاسبه class weights
        self.calculate_class_weights(y_entry_balanced, y_exit_balanced)
        
        return {
            'X_train': X_train_balanced,
            'X_test': X_test_scaled,
            'y_entry_train': y_entry_balanced,
            'y_entry_test': y_entry_test,
            'y_exit_train': y_exit_balanced,
            'y_exit_test': y_exit_test
        }
    
    def find_optimal_threshold(self, X_val, y_entry_val, y_exit_val):
        """
        یافتن بهترین threshold بر اساس F1-score
        """
        print("🎯 جستجوی بهترین threshold...")
        
        # پیش‌بینی احتمالات
        entry_probs, exit_probs = self.model.predict(X_val, verbose=0)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_entry_threshold = 0.5
        best_exit_threshold = 0.5
        best_entry_f1 = 0
        best_exit_f1 = 0
        
        # Entry threshold optimization
        for threshold in thresholds:
            entry_pred = (entry_probs > threshold).astype(int)
            f1 = f1_score(y_entry_val, entry_pred)
            if f1 > best_entry_f1:
                best_entry_f1 = f1
                best_entry_threshold = threshold
        
        # Exit threshold optimization  
        for threshold in thresholds:
            exit_pred = (exit_probs > threshold).astype(int)
            f1 = f1_score(y_exit_val, exit_pred)
            if f1 > best_exit_f1:
                best_exit_f1 = f1
                best_exit_threshold = threshold
                
        self.threshold_entry = best_entry_threshold
        self.threshold_exit = best_exit_threshold
        
        print(f"✅ بهترین Entry Threshold: {best_entry_threshold:.3f} (F1: {best_entry_f1:.3f})")
        print(f"✅ بهترین Exit Threshold: {best_exit_threshold:.3f} (F1: {best_exit_f1:.3f})")
    
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
                'best_improved_model.h5', save_best_only=True, monitor='val_loss', verbose=1
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
    # ایجاد و آموزش مدل بهبود یافته
    model = ImprovedTradingModel(use_focal_loss=True)
    
    print("🎯 شروع آموزش مدل بهبود یافته با حل مشکل Class Imbalance")
    print("="*60)
    
    history = model.train(
        training_file='training_data.csv',
        epochs=2,
        batch_size=512
    )
    
    print("\n✅ آموزش کامل شد!")
    print(f"🎯 Entry Threshold: {model.threshold_entry:.3f}")
    print(f"🎯 Exit Threshold: {model.threshold_exit:.3f}") 