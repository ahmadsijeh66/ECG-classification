import os
import wfdb
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Bidirectional, LSTM, Attention, Dropout, BatchNormalization, GlobalAveragePooling1D, Layer
from keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW # يتطلب تثبيت tensorflow-addons
#  (Attention layer)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

#  Keras Backend
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical
import gc
from tqdm import tqdm
from collections import Counter

# Configuration optimized for 12GB RAM
BASE_PATH = r"C:\Users\ahmad\Desktop\البحث التطبيقي\dataset\mit-bih-arrhythmia-database-1.0.0"
MODEL_PATH = 'ecg_classifier_mitbih_enhanced.h5'
SAMPLE_LENGTH = 720
NUM_LEADS = 2
MAX_RECORDS = 48
CHUNK_SIZE = 2500
BATCH_SIZE = 32
SAMPLING_RATE = 360  # MIT-BIH sampling rate is 360 Hz

# AAMI recommended class groupings
AAMI_CLASSES = {
    'N': ['N'],  # Normal beats
    'L': ['L'],
    'R': ['R'],
    'S': ['S','a','A','J'],  # Supraventricular beats
    'V': ['V'],  # Ventricular beats
    'F': ['F'],  # Fusion beats
    'Q': ['Q', '/', 'f', 'U']  # Unknown beats
}

class ECGDigitalFilters:
    """Digital filters for ECG signal preprocessing"""
    
    @staticmethod
    def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=SAMPLING_RATE, order=4):
        """Bandpass filter to remove baseline wander and high-frequency noise"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, signal)
    
    @staticmethod
    def notch_filter(signal, freq=60.0, Q=30.0, fs=SAMPLING_RATE):
        """Notch filter to remove power line interference (60Hz/50Hz)"""
        nyquist = 0.5 * fs
        freq_normalized = freq / nyquist
        b, a = scipy_signal.iirnotch(freq_normalized, Q)
        return scipy_signal.filtfilt(b, a, signal)
    
    @staticmethod
    def highpass_filter(signal, cutoff=0.5, fs=SAMPLING_RATE, order=4):
        """Highpass filter to remove baseline wander"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = scipy_signal.butter(order, normal_cutoff, btype='high')
        return scipy_signal.filtfilt(b, a, signal)
    
    @staticmethod
    def lowpass_filter(signal, cutoff=40.0, fs=SAMPLING_RATE, order=4):
        """Lowpass filter to remove high-frequency noise"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = scipy_signal.butter(order, normal_cutoff, btype='low')
        return scipy_signal.filtfilt(b, a, signal)
    
    @staticmethod
    def apply_filters(signal, filter_types=['bandpass', 'notch']):
        """Apply multiple filters to ECG signal"""
        filtered_signal = signal.copy()
        
        for filter_type in filter_types:
            if filter_type == 'bandpass':
                for lead in range(signal.shape[1]):
                    filtered_signal[:, lead] = ECGDigitalFilters.bandpass_filter(filtered_signal[:, lead])
            elif filter_type == 'notch':
                for lead in range(signal.shape[1]):
                    filtered_signal[:, lead] = ECGDigitalFilters.notch_filter(filtered_signal[:, lead])
            elif filter_type == 'highpass':
                for lead in range(signal.shape[1]):
                    filtered_signal[:, lead] = ECGDigitalFilters.highpass_filter(filtered_signal[:, lead])
            elif filter_type == 'lowpass':
                for lead in range(signal.shape[1]):
                    filtered_signal[:, lead] = ECGDigitalFilters.lowpass_filter(filtered_signal[:, lead])
        
        return filtered_signal

def load_mitbih_annotations(record_path):
    """Load annotations from MIT-BIH record"""
    try:
        annotation = wfdb.rdann(record_path, 'atr')
        return annotation.symbol, annotation.sample
    except Exception as e:
        print(f"Error loading annotations for {record_path}: {str(e)}")
        return None, None

def extract_beats_from_record(record_path, target_classes, apply_filter=True):
    """Extract beats from a single MIT-BIH record with optional filtering"""
    try:
        # Read the record
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal.T.astype(np.float32)  # Shape: (leads, samples)
        
        # Apply digital filters if requested
        if apply_filter:
            signals_filtered = np.empty_like(signals)
            for lead in range(signals.shape[0]):
                # Apply bandpass and notch filters
                filtered_lead = ECGDigitalFilters.bandpass_filter(signals[lead])
                filtered_lead = ECGDigitalFilters.notch_filter(filtered_lead)
                signals_filtered[lead] = filtered_lead
            signals = signals_filtered
        
        # Load annotations
        symbols, samples = load_mitbih_annotations(record_path)
        if symbols is None:
            return [], []
        
        beats = []
        labels = []
        
        # Extract beats around each annotation
        for symbol, sample in zip(symbols, samples):
            # Map to AAMI classes
            aami_class = None
            for aami_key, mitbih_symbols in AAMI_CLASSES.items():
                if symbol in mitbih_symbols:
                    aami_class = aami_key
                    break
            
            # Only keep beats from our target classes
            if aami_class in target_classes:
                # Extract beat segment
                start_idx = max(0, sample - SAMPLE_LENGTH // 2)
                end_idx = min(signals.shape[1], sample + SAMPLE_LENGTH // 2)
                
                if end_idx - start_idx >= SAMPLE_LENGTH:
                    # Extract the beat
                    beat_segment = signals[:, start_idx:end_idx]
                    
                    # If segment is longer than needed, take center
                    if beat_segment.shape[1] > SAMPLE_LENGTH:
                        start_trim = (beat_segment.shape[1] - SAMPLE_LENGTH) // 2
                        beat_segment = beat_segment[:, start_trim:start_trim + SAMPLE_LENGTH]
                    
                    beats.append(beat_segment.T)  # Transpose to (samples, leads)
                    labels.append(aami_class)
        
        return beats, labels
        
    except Exception as e:
        print(f"Error processing record {record_path}: {str(e)}")
        return [], []

def find_mitbih_records(root_folder):
    """Find all MIT-BIH record files"""
    record_files = []
    for root, _, files in tqdm(os.walk(root_folder), desc="Scanning MIT-BIH records"):
        # Look for .dat files (signal files)
        dat_files = [f for f in files if f.endswith('.dat')]
        for dat_file in dat_files:
            record_name = dat_file[:-4]  # Remove .dat extension
            record_path = os.path.join(root, record_name)
            
            # Check if corresponding .atr (annotation) file exists
            atr_file = record_name + '.atr'
            if atr_file in files:
                record_files.append(record_path)
            
            if len(record_files) >= MAX_RECORDS:
                break
                
        if len(record_files) >= MAX_RECORDS:
            break
            
    return record_files

def process_mitbih_chunk(chunk_records, target_classes, apply_filter=True):
    """Process a chunk of MIT-BIH records"""
    X_chunk = []
    y_chunk = []
    ids_chunk = []
    
    for record_path in chunk_records:
        beats, labels = extract_beats_from_record(record_path, target_classes, apply_filter)
        
        if beats and labels:
            X_chunk.extend(beats)
            y_chunk.extend(labels)
            ids_chunk.extend([os.path.basename(record_path)] * len(beats))
    
    return X_chunk, y_chunk, ids_chunk

def load_mitbih_dataset(target_classes, apply_filter=True):
    """Load MIT-BIH dataset with memory optimization and optional filtering"""
    record_files = find_mitbih_records(BASE_PATH)
    if not record_files:
        print("No MIT-BIH records found")
        return None, None, None
    
    X_final = []
    y_final = []
    ids_final = []
    
    print(f"Processing {len(record_files)} MIT-BIH records in chunks...")
    print(f"Digital filtering: {'ENABLED' if apply_filter else 'DISABLED'}")
    
    for i in tqdm(range(0, len(record_files), CHUNK_SIZE), desc="Processing"):
        chunk = record_files[i:i + CHUNK_SIZE]
        X_chunk, y_chunk, ids_chunk = process_mitbih_chunk(chunk, target_classes, apply_filter)
        
        if X_chunk:
            X_final.extend(X_chunk)
            y_final.extend(y_chunk)
            ids_final.extend(ids_chunk)
        
        # Memory cleanup
        del X_chunk, y_chunk, ids_chunk
        gc.collect()
    
    return np.array(X_final), np.array(y_final), ids_final

def preprocess_data(X, y, normalization_method='standard'):
    """Memory-efficient preprocessing with different normalization options"""
    X_normalized = np.empty_like(X, dtype=np.float32)
    
    for i in tqdm(range(X.shape[0]), desc=f"Normalizing ({normalization_method})"):
        for lead in range(X.shape[2]):
            signal = X[i, :, lead]
            
            if normalization_method == 'standard':
                # Standard normalization (zero mean, unit variance)
                X_normalized[i, :, lead] = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            elif normalization_method == 'minmax':
                # Min-Max normalization to [0, 1]
                X_normalized[i, :, lead] = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)
            
            elif normalization_method == 'robust':
                # Robust normalization using median and IQR
                median = np.median(signal)
                iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
                X_normalized[i, :, lead] = (signal - median) / (iqr + 1e-8)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Print class mapping for debugging
    print("\nLabel Encoder Class Mapping:")
    for i, class_name in enumerate(le.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {i} -> {class_name}: {count:,} samples")
    
    return X_normalized, y_categorical, le

def compute_enhanced_class_weights(y):
    """Enhanced class weighting with multiple strategies"""
    y_flat = np.argmax(y, axis=1)
    class_counts = Counter(y_flat)
    total_samples = len(y_flat)
    num_classes = len(class_counts)
    
    print(f"Class distribution: {class_counts}")
    
    # Strategy 1: Standard balanced weights
    balanced_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_flat),
        y=y_flat
    )
    
    # Strategy 2: Inverse frequency with smoothing
    inverse_freq_weights = total_samples / (num_classes * np.array([class_counts[i] for i in range(num_classes)]))
    
    # Strategy 3: Logarithmic weighting (less aggressive)
    log_weights = np.log(total_samples / np.array([class_counts[i] for i in range(num_classes)]))
    log_weights = log_weights / np.min(log_weights)  # Normalize
    
    # Strategy 4: Combined approach (balanced + logarithmic)
    combined_weights = (balanced_weights + log_weights) / 2
    
    # Choose the best strategy based on class distribution
    min_samples = min(class_counts.values())
    max_samples = max(class_counts.values())
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    if imbalance_ratio > 100:  # Highly imbalanced
        selected_weights = log_weights
        print("Using logarithmic weighting for highly imbalanced data")
    elif imbalance_ratio > 10:  # Moderately imbalanced
        selected_weights = combined_weights
        print("Using combined weighting for moderately imbalanced data")
    else:  # Relatively balanced
        selected_weights = balanced_weights
        print("Using standard balanced weighting")
    
    class_weight_dict = {i: weight for i, weight in enumerate(selected_weights)}
    print(f"Enhanced class weights: {class_weight_dict}")
    
    return class_weight_dict

def handle_class_imbalance(X, y, strategy='class_weight'):
    """
    Handle class imbalance using different strategies
    Options: 'class_weight', 'undersample', 'oversample', 'smote', 'combined'
    """
    y_flat = np.argmax(y, axis=1)
    original_dist = Counter(y_flat)
    print(f"Original class distribution: {original_dist}")
    
    if strategy == 'class_weight':
        # Use enhanced class weights
        class_weight_dict = compute_enhanced_class_weights(y)
        return X, y, class_weight_dict
    
    elif strategy == 'undersample':
        # Random undersampling
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_resampled, y_resampled = rus.fit_resample(X_reshaped, y_flat)
        X_resampled = X_resampled.reshape(-1, SAMPLE_LENGTH, NUM_LEADS)
        print(f"After undersampling: {Counter(y_resampled)}")
        return X_resampled, to_categorical(y_resampled), None
    
    elif strategy == 'oversample':
        # Random oversampling
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_resampled, y_resampled = ros.fit_resample(X_reshaped, y_flat)
        X_resampled = X_resampled.reshape(-1, SAMPLE_LENGTH, NUM_LEADS)
        print(f"After oversampling: {Counter(y_resampled)}")
        return X_resampled, to_categorical(y_resampled), None
    
    elif strategy == 'smote':
        # SMOTE oversampling
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_flat)
        X_resampled = X_resampled.reshape(-1, SAMPLE_LENGTH, NUM_LEADS)
        print(f"After SMOTE: {Counter(y_resampled)}")
        return X_resampled, to_categorical(y_resampled), None
    
    elif strategy == 'combined':
        # Combined SMOTE + Undersampling
        from imblearn.combine import SMOTEENN
        smote_enn = SMOTEENN(random_state=42)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_resampled, y_resampled = smote_enn.fit_resample(X_reshaped, y_flat)
        X_resampled = X_resampled.reshape(-1, SAMPLE_LENGTH, NUM_LEADS)
        print(f"After SMOTEENN: {Counter(y_resampled)}")
        return X_resampled, to_categorical(y_resampled), None
    
    else:
        return X, y, None

def plot_class_distribution(y, le=None, title="Class Distribution"):
    """Plot class distribution with correct class labels"""
    if le is None:
        # If no label encoder provided, assume y is already encoded
        y_flat = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        class_counts = Counter(y_flat)
        # Use default class order based on the actual data
        unique_classes = sorted(class_counts.keys())
        classes = [f'Class_{i}' for i in unique_classes]
    else:
        # Use the label encoder to get correct class order
        y_flat = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        class_counts = Counter(y_flat)
        classes = le.classes_
    
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'cyan', 'lightblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    
    # Create bars for each class in the correct order
    bar_values = [class_counts.get(i, 0) for i in range(len(classes))]
    bars = plt.bar(range(len(classes)), bar_values, color=colors[:len(classes)])
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Beat Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, bar_values):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bar_values)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print actual distribution
    print(f"\n{title}:")
    total_samples = sum(bar_values)
    for i, class_name in enumerate(classes):
        count = class_counts.get(i, 0)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {class_name}: {count:,} samples ({percentage:.2f}%)")

def debug_class_distribution(y, le):
    """Debug function to see actual class distribution"""
    y_flat = np.argmax(y, axis=1) if len(y.shape) > 1 else y
    class_counts = Counter(y_flat)
    
    print("\nDEBUG - Class Distribution:")
    print("Encoded labels -> Class names:")
    for encoded_label, class_name in enumerate(le.classes_):
        count = class_counts.get(encoded_label, 0)
        print(f"  {encoded_label} -> {class_name}: {count:,} samples")
    
    # Check which class has the most samples
    if class_counts:
        most_common_class_num = max(class_counts, key=class_counts.get)
        most_common_class_name = le.classes_[most_common_class_num]
        print(f"\nMost common class: {most_common_class_name} (label {most_common_class_num}) with {class_counts[most_common_class_num]:,} samples")
    
    return class_counts

def build_bilstm_cnn_model(input_shape, num_classes):
    """Hybrid BiLSTM-CNN model for ECG classification"""
    model = Sequential([
        # First convolutional block
        Conv1D(64, 7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3), # زيادة Dropout
        
        # Second convolutional block
        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3), # زيادة Dropout
        
        # Third convolutional block
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3), # زيادة Dropout
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3), # زيادة Dropout
        
        Bidirectional(LSTM(128, return_sequences=True)), # تغيير إلى True لإضافة طبقة الانتباه
        BatchNormalization(),
        Dropout(0.3), # زيادة Dropout
        
        # إضافة طبقة الانتباه (Attention)
        AttentionLayer(), # استخدام طبقة الانتباه المخصصة
        
        # طبقة Dense النهائية
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # استخدام AdamW مع معدل تعلم أولي منخفض
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_cnn_model(input_shape, num_classes):
    """CNN-only model for comparison"""
    model = Sequential([
        Conv1D(64, 7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        
        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        
        Conv1D(512, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        
        GlobalAveragePooling1D(),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # استخدام AdamW مع معدل تعلم أولي منخفض
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history, model_name):
    """Enhanced training history visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='green')
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        # Create empty subplot if learning rate not available
        axes[1, 0].text(0.5, 0.5, 'Learning Rate Data\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
    
    # Accuracy difference
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    acc_diff = [train_acc[i] - val_acc[i] for i in range(len(train_acc))]
    axes[1, 1].plot(acc_diff, label='Train-Val Accuracy Difference', linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Train-Validation Accuracy Gap', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_evaluation(y_true, y_pred, class_names, history, method_name):
    """Create comprehensive evaluation plots"""
    fig, axes = plt.subplots(2, 3, figsize=(25, 16))
    fig.suptitle(f'Evaluation Results - {method_name}', fontsize=16, fontweight='bold')
    
    # 1. Training History - Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training History - Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0, 2])
    axes[0, 2].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Predicted Label')
    axes[0, 2].set_ylabel('True Label')
    
    # 4. Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    
    # 5. Class-wise Performance
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    precision = [report[cls]['precision'] for cls in class_names]
    recall = [report[cls]['recall'] for cls in class_names]
    f1 = [report[cls]['f1-score'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[1, 1].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[1, 1].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[1, 1].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    axes[1, 1].set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Training Progress (Last 10 epochs zoom)
    if len(history.history['accuracy']) > 10:
        last_epochs = range(len(history.history['accuracy']) - 10, len(history.history['accuracy']))
        axes[1, 2].plot(last_epochs, history.history['accuracy'][-10:], label='Training Accuracy', marker='o')
        axes[1, 2].plot(last_epochs, history.history['val_accuracy'][-10:], label='Validation Accuracy', marker='s')
        axes[1, 2].set_title('Last 10 Epochs (Zoom)', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Create empty subplot if not enough epochs
        axes[1, 2].text(0.5, 0.5, 'Not enough epochs\nfor zoom view', 
                       ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Last 10 Epochs (Zoom)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'evaluation_{method_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_with_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics"""
    # Overall metrics
    accuracy = np.mean(y_true == y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE MODEL EVALUATION")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print(f"{'='*70}")
    
    # Per-class performance
    print("\nPer-Class Performance:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name:<10} {metrics['precision']:.4f}     {metrics['recall']:.4f}      {metrics['f1-score']:.4f}      {metrics['support']:<10}")
    
    return report, accuracy, f1_weighted

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, le, class_weights=None, method_name="Baseline", model_type='bilstm_cnn'):
    """Train and evaluate model with given data"""
    print(f"\n{'='*60}")
    print(f"TRAINING WITH METHOD: {method_name}")
    print(f"MODEL TYPE: {model_type}")
    print(f"{'='*60}")
    
    # Build model
    if model_type == 'bilstm_cnn':
        model = build_bilstm_cnn_model((SAMPLE_LENGTH, NUM_LEADS), len(le.classes_))
    else:
        model = build_cnn_model((SAMPLE_LENGTH, NUM_LEADS), len(le.classes_))
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy', verbose=1),
        ModelCheckpoint(f'model_{method_name}_{model_type}.h5', save_best_only=True, monitor='val_accuracy'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1) # تقليل عامل التخفيض والصبر
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, f"{method_name}_{model_type}")
    
    # Evaluate model
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nResults for {method_name} ({model_type}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    # Plot evaluation
    plot_detailed_evaluation(y_true, y_pred, le.classes_, history, f"{method_name}_{model_type}")
    
    return model, history, accuracy, f1_weighted, f1_macro, y_true, y_pred

# Simple training function for quick testing
def train_simple_model():
    """Simple training function for quick testing with BiLSTM"""
    print("Loading MIT-BIH dataset...")
    X, y, _ = load_mitbih_dataset(list(AAMI_CLASSES.keys()), apply_filter=True)
    
    if X is None:
        print("No data found!")
        return
    
    print(f"Loaded {len(X)} beats")
    
    # Preprocess
    X_processed, y_processed, le = preprocess_data(X, y)
    
    # Plot original class distribution with correct labels
    plot_class_distribution(y_processed, le, "Original Class Distribution")
    
    # Debug class distribution
    debug_class_distribution(y_processed, le)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Handle imbalance with enhanced class weighting
    X_balanced, y_balanced, class_weights = handle_class_imbalance(
        X_train, y_train, strategy='class_weight'
    )
    
    # Train BiLSTM-CNN model
    print("Training BiLSTM-CNN model with enhanced class weighting...")
    model = build_bilstm_cnn_model((SAMPLE_LENGTH, NUM_LEADS), len(le.classes_))
    
    history = model.fit(
        X_balanced, y_balanced,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'), # زيادة الصبر والمراقبة على الخسارة
            ModelCheckpoint('best_bilstm_model_enhanced.h5', save_best_only=True, monitor='val_loss'), # المراقبة على الخسارة
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7) # تقليل عامل التخفيض والصبر
        ],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, "BiLSTM_CNN_Enhanced")
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    accuracy = np.mean(y_true == y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print(f"BiLSTM-CNN Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    
    # Plot detailed evaluation
    plot_detailed_evaluation(y_true, y_pred, le.classes_, history, "BiLSTM_CNN_Final")
    
    return model, history

if __name__ == "__main__":
    # Run simple training for quick testing with BiLSTM
    train_simple_model()
