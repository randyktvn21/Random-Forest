import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Class labels
CLASS_LABELS = {
    1: 'glass',    # Changed to lowercase
    2: 'paper',
    3: 'cardboard',
    4: 'plastic',
    5: 'metal',
    6: 'trash'
}

def load_dataset(train_file, val_file, test_file):
    """Load dataset from the text files"""
    train_data = pd.read_csv(train_file, sep=' ', header=None, names=['filename', 'label'])
    val_data = pd.read_csv(val_file, sep=' ', header=None, names=['filename', 'label'])
    test_data = pd.read_csv(test_file, sep=' ', header=None, names=['filename', 'label'])
    return train_data, val_data, test_data

def load_and_preprocess_image(image_path, target_size=(128, 128)):  # Increased image size
    """Load and preprocess a single image"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
            
        # Resize to fixed size
        img = cv2.resize(img, target_size)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Additional preprocessing steps
        # 1. Contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Normalize pixel values
        img_normalized = enhanced.astype('float32') / 255.0
        
        # 3. Extract color features
        color_features = np.mean(img_normalized, axis=(0,1))  # Mean RGB values
        
        # 4. Extract texture features (using standard deviation of each channel)
        texture_features = np.std(img_normalized, axis=(0,1))
        
        # Combine features
        features = np.concatenate([
            img_normalized.reshape(-1),  # Flattened image
            color_features,              # Color features
            texture_features             # Texture features
        ])
        
        return features
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_data(data_df, base_path):
    """Prepare features and labels from dataframe"""
    X = []
    y = []
    failed_images = 0
    
    for idx, row in data_df.iterrows():
        # Get category name directly from CLASS_LABELS
        category = CLASS_LABELS[row['label']]  # This will already be lowercase
        
        # Construct the image path
        img_path = os.path.join(base_path, category, row['filename'])
        
        if os.path.exists(img_path):
            img_features = load_and_preprocess_image(img_path)
            if img_features is not None:
                X.append(img_features)
                y.append(row['label'])
            else:
                failed_images += 1
        else:
            print(f"Image not found: {img_path}")
            failed_images += 1
        
        # Print progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} images")
            print(f"Current dataset size: {len(X)} images")
            print(f"Failed images: {failed_images}")
    
    if len(X) == 0:
        raise ValueError("No images were successfully loaded. Check the image paths and data directory structure.")
    
    return np.array(X), np.array(y)

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("\nTraining Random Forest model...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Initialize model with better parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Evaluate on training data
    train_score = rf_model.score(X_train, y_train)
    print(f"\nTraining accuracy: {train_score:.4f}")
    
    # Evaluate on validation data
    val_score = rf_model.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Get predictions
    y_pred = rf_model.predict(X_val)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=list(CLASS_LABELS.values())))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS.values(),
                yticklabels=CLASS_LABELS.values())
    plt.title('Confusion Matrix - Garbage Classification')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('static/confusion_matrix.png')
    plt.close()
    
    # Feature Importance Analysis
    if hasattr(rf_model, 'feature_importances_'):
        # Calculate feature importance scores
        importances = rf_model.feature_importances_
        
        # Group features by type
        n_color_features = 3  # RGB channels
        n_texture_features = 3  # Standard deviation of RGB channels
        n_pixel_features = len(importances) - n_color_features - n_texture_features
        
        # Calculate average importance for each feature type
        pixel_importance = np.mean(importances[:n_pixel_features])
        color_importance = np.mean(importances[n_pixel_features:n_pixel_features+n_color_features])
        texture_importance = np.mean(importances[-n_texture_features:])
        
        # Create feature importance summary
        feature_types = ['Image Pixels', 'Color (RGB)', 'Texture']
        feature_importances = [pixel_importance, color_importance, texture_importance]
        
        # Plot feature type importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Type Importance in Garbage Classification")
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = plt.bar(feature_types, feature_importances, color=colors)
        plt.xlabel('Feature Type')
        plt.ylabel('Average Importance')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('static/feature_importance.png')
        plt.close()
        
        # Print feature importance summary
        print("\nFeature Importance Summary:")
        for feat_type, importance in zip(feature_types, feature_importances):
            print(f"{feat_type}: {importance:.4f}")
    
    return rf_model

def main():
    # Base path to your dataset
    base_path = "Garbage classification/Garbage classification"
    
    # Verify base path exists
    if not os.path.exists(base_path):
        print(f"Error: Directory not found: {base_path}")
        print("Current working directory:", os.getcwd())
        return
    
    # List contents of base directory
    print("Contents of base directory:")
    print(os.listdir(base_path))
    
    # Load dataset information
    print("\nLoading dataset information...")
    try:
        train_data, val_data, test_data = load_dataset(
            'one-indexed-files-notrash_train.txt',
            'one-indexed-files-notrash_val.txt',
            'one-indexed-files-notrash_test.txt'
        )
    except Exception as e:
        print(f"Error loading dataset files: {str(e)}")
        return
    
    try:
        # Prepare training data
        print("\nPreparing training data...")
        X_train, y_train = prepare_data(train_data, base_path)
        
        # Prepare validation data
        print("\nPreparing validation data...")
        X_val, y_val = prepare_data(val_data, base_path)
        
        # Train model
        model = train_random_forest(X_train, y_train, X_val, y_val)
        
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save the model
        print("\nSaving model...")
        model_path = "model/random_forest_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 