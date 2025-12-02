"""
Comprehensive Evaluation Script for Human Activity Recognition Model
Generates all evaluation metrics, visualizations, and saves results to result.txt
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_and_classes():
    """Load the trained model and class names"""
    model_path = 'activity_recognition_model.keras'
    class_names_path = 'class_names.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return None, None
    
    if not os.path.exists(class_names_path):
        print(f"Error: Class names file '{class_names_path}' not found!")
        return None, None
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    class_names = joblib.load(class_names_path)
    
    print(f"Model loaded successfully!")
    print(f"Number of classes: {len(class_names)}")
    return model, class_names

def evaluate_model(model, class_names):
    """Comprehensive evaluation on validation set"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Load training data for validation split
    train_data = pd.read_csv("Dataset/Training_set.csv")
    train_folder = "Dataset/train"
    
    # Create validation generator (same as training)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory=train_folder,
        x_col="filename",
        y_col="label",
        target_size=(160, 160),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"\nValidation samples: {val_gen.samples}")
    
    # Get predictions
    val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    print("\nGenerating predictions...")
    val_predictions = model.predict(val_gen, steps=val_steps, verbose=1)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_true_classes = val_gen.classes
    confidence_scores = np.max(val_predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(val_true_classes, val_pred_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        val_true_classes, val_pred_classes, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        val_true_classes, val_pred_classes, average='weighted', zero_division=0
    )
    
    # Additional metrics
    kappa = cohen_kappa_score(val_true_classes, val_pred_classes)
    mcc = matthews_corrcoef(val_true_classes, val_pred_classes)
    
    # Confusion matrix
    cm = confusion_matrix(val_true_classes, val_pred_classes)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    
    # Generate visualizations
    generate_visualizations(cm, class_names, confidence_scores, 
                           val_true_classes, val_pred_classes, 
                           per_class_accuracy)
    
    # Save results to file
    save_results_to_file(
        class_names, accuracy, precision, recall, f1, support,
        macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1,
        kappa, mcc, cm, per_class_accuracy, confidence_scores
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'kappa': kappa,
        'mcc': mcc,
        'cm': cm,
        'per_class_accuracy': per_class_accuracy,
        'confidence_scores': confidence_scores
    }

def generate_visualizations(cm, class_names, confidence_scores, 
                           val_true_classes, val_pred_classes, 
                           per_class_accuracy):
    """Generate all evaluation visualizations"""
    print("\nGenerating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Validation Set', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  - Saved: confusion_matrix.png")
    plt.close()
    
    # 2. Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    plt.title('Normalized Confusion Matrix - Validation Set', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print("  - Saved: confusion_matrix_normalized.png")
    plt.close()
    
    # 3. Per-Class Accuracy
    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(class_names)), per_class_accuracy, color='steelblue', alpha=0.7)
    plt.xlabel('Activity Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy - Validation Set', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("  - Saved: per_class_accuracy.png")
    plt.close()
    
    # 4. Precision, Recall, F1-Score per Class
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_true_classes, val_pred_classes, average=None, zero_division=0
    )
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Activity Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, and F1-Score per Class - Validation Set', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('precision_recall_f1_per_class.png', dpi=300, bbox_inches='tight')
    print("  - Saved: precision_recall_f1_per_class.png")
    plt.close()
    
    # 5. Confidence Score Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(confidence_scores, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Confidence Scores - Validation Set', 
              fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidence_scores):.3f}')
    plt.axvline(np.median(confidence_scores), color='green', linestyle='--', 
                label=f'Median: {np.median(confidence_scores):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("  - Saved: confidence_distribution.png")
    plt.close()
    
    # 6. Confidence by Class
    confidence_by_class = []
    for i in range(len(class_names)):
        mask = val_pred_classes == i
        if np.sum(mask) > 0:
            confidence_by_class.append(np.mean(confidence_scores[mask]))
        else:
            confidence_by_class.append(0.0)
    
    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(class_names)), confidence_by_class, color='coral', alpha=0.7)
    plt.xlabel('Activity Class', fontsize=12)
    plt.ylabel('Mean Confidence Score', fontsize=12)
    plt.title('Mean Prediction Confidence by Class - Validation Set', 
              fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, conf) in enumerate(zip(bars, confidence_by_class)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('confidence_by_class.png', dpi=300, bbox_inches='tight')
    print("  - Saved: confidence_by_class.png")
    plt.close()
    
    print("\nAll visualizations generated successfully!")

def save_results_to_file(class_names, accuracy, precision, recall, f1, support,
                         macro_precision, macro_recall, macro_f1,
                         weighted_precision, weighted_recall, weighted_f1,
                         kappa, mcc, cm, per_class_accuracy, confidence_scores):
    """Save all evaluation results to result.txt in table format"""
    
    print("\nSaving results to result.txt...")
    
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HUMAN ACTIVITY RECOGNITION MODEL - COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: EfficientNet-B3\n")
        f.write(f"Dataset: Validation Set (20% split from training data)\n")
        f.write(f"Total Classes: {len(class_names)}\n")
        f.write(f"Total Samples: {len(support)}\n")
        f.write("="*80 + "\n\n")
        
        # Overall Metrics
        f.write("OVERALL MODEL PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<30} {'Value':<20} {'Percentage':<20}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Accuracy':<30} {accuracy:<20.6f} {accuracy*100:<20.2f}%\n")
        f.write(f"{'Macro Average Precision':<30} {macro_precision:<20.6f} {macro_precision*100:<20.2f}%\n")
        f.write(f"{'Macro Average Recall':<30} {macro_recall:<20.6f} {macro_recall*100:<20.2f}%\n")
        f.write(f"{'Macro Average F1-Score':<30} {macro_f1:<20.6f} {macro_f1*100:<20.2f}%\n")
        f.write(f"{'Weighted Average Precision':<30} {weighted_precision:<20.6f} {weighted_precision*100:<20.2f}%\n")
        f.write(f"{'Weighted Average Recall':<30} {weighted_recall:<20.6f} {weighted_recall*100:<20.2f}%\n")
        f.write(f"{'Weighted Average F1-Score':<30} {weighted_f1:<20.6f} {weighted_f1*100:<20.2f}%\n")
        f.write(f"{'Cohen Kappa Score':<30} {kappa:<20.6f} {'N/A':<20}\n")
        f.write(f"{'Matthews Correlation Coefficient':<30} {mcc:<20.6f} {'N/A':<20}\n")
        f.write(f"{'Mean Confidence Score':<30} {np.mean(confidence_scores):<20.6f} {np.mean(confidence_scores)*100:<20.2f}%\n")
        f.write(f"{'Median Confidence Score':<30} {np.median(confidence_scores):<20.6f} {np.median(confidence_scores)*100:<20.2f}%\n")
        f.write(f"{'Std Confidence Score':<30} {np.std(confidence_scores):<20.6f} {'N/A':<20}\n")
        f.write("\n")
        
        # Per-Class Metrics Table
        f.write("PER-CLASS PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<25} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                   f"{f1[i]:<12.4f} {per_class_accuracy[i]:<12.4f} {int(support[i]):<10}\n")
        
        f.write("\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        f.write(f"{'True\\Pred':<20}")
        for class_name in class_names:
            f.write(f"{class_name[:10]:<12}")
        f.write("\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name[:18]:<20}")
            for j in range(len(class_names)):
                f.write(f"{cm[i, j]:<12}")
            f.write("\n")
        
        f.write("\n")
        
        # Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        f.write("NORMALIZED CONFUSION MATRIX (Row Normalized)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'True\\Pred':<20}")
        for class_name in class_names:
            f.write(f"{class_name[:10]:<12}")
        f.write("\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name[:18]:<20}")
            for j in range(len(class_names)):
                f.write(f"{cm_normalized[i, j]:<12.4f}")
            f.write("\n")
        
        f.write("\n")
        
        # Confidence Statistics by Class
        f.write("CONFIDENCE STATISTICS BY CLASS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<25} {'Mean Confidence':<18} {'Std Confidence':<18} {'Min Confidence':<18} {'Max Confidence':<18}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            mask = val_pred_classes == i
            if np.sum(mask) > 0:
                class_confidences = confidence_scores[mask]
                f.write(f"{class_name:<25} {np.mean(class_confidences):<18.4f} "
                       f"{np.std(class_confidences):<18.4f} {np.min(class_confidences):<18.4f} "
                       f"{np.max(class_confidences):<18.4f}\n")
            else:
                f.write(f"{class_name:<25} {'0.0000':<18} {'0.0000':<18} {'0.0000':<18} {'0.0000':<18}\n")
        
        f.write("\n")
        
        # Summary Statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Validation Samples: {np.sum(support)}\n")
        f.write(f"Correct Predictions: {np.sum(cm.diagonal())}\n")
        f.write(f"Incorrect Predictions: {np.sum(support) - np.sum(cm.diagonal())}\n")
        f.write(f"Classes with Perfect Accuracy: {np.sum(per_class_accuracy == 1.0)}\n")
        f.write(f"Classes with Zero Accuracy: {np.sum(per_class_accuracy == 0.0)}\n")
        f.write(f"Best Performing Class: {class_names[np.argmax(per_class_accuracy)]} "
               f"(Accuracy: {np.max(per_class_accuracy):.4f})\n")
        f.write(f"Worst Performing Class: {class_names[np.argmin(per_class_accuracy)]} "
               f"(Accuracy: {np.min(per_class_accuracy):.4f})\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF EVALUATION REPORT\n")
        f.write("="*80 + "\n")
    
    print("  - Results saved to: result.txt")

def main():
    """Main evaluation function"""
    print("="*70)
    print("HUMAN ACTIVITY RECOGNITION - MODEL EVALUATION")
    print("="*70)
    
    # Load model
    model, class_names = load_model_and_classes()
    if model is None:
        return
    
    # Evaluate model
    results = evaluate_model(model, class_names)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Macro F1-Score: {results['macro_f1']:.4f} ({results['macro_f1']*100:.2f}%)")
    print(f"Weighted F1-Score: {results['weighted_f1']:.4f} ({results['weighted_f1']*100:.2f}%)")
    print(f"\nResults saved to: result.txt")
    print(f"Visualizations saved as PNG files in current directory")

if __name__ == "__main__":
    main()

