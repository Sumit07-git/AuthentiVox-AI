"""
Quick Start Script
Automates the setup and training process
"""

import os
import sys
import subprocess


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        DeepGuard AI - Quick Start Automation             ║
    ║        Hybrid Deepfake Audio Detection System            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\nThis script will:")
    print("1. Generate sample training data (if needed)")
    print("2. Train Machine Learning model")
    print("3. Train Deep Learning model")
    print("4. Start Flask application")
    
    # Check if data exists
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("\n⚠ Training data not found!")
        choice = input("\nGenerate sample data? (y/n): ").lower()
        
        if choice == 'y':
            num_samples = input("Number of samples per class (default 50): ") or "50"
            run_command(
                f"python generate_sample_data.py --samples {num_samples}",
                "Generating sample dataset"
            )
        else:
            print("\n❌ Cannot proceed without training data.")
            print("Please add your audio files to:")
            print(f"  - {real_dir}")
            print(f"  - {fake_dir}")
            sys.exit(1)
    else:
        real_files = len([f for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3'))])
        fake_files = len([f for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3'))])
        print(f"\n✓ Found {real_files} real and {fake_files} fake audio files")
    
    # Check if models exist
    ml_model_exists = os.path.exists('models/ml_model/rf_classifier.pkl')
    dl_model_exists = os.path.exists('models/dl_model/cnn_model.keras')
    
    if ml_model_exists and dl_model_exists:
        print("\n✓ Models already trained!")
        choice = input("Retrain models? (y/n): ").lower()
        if choice != 'y':
            print("\nSkipping training, starting application...")
            run_command("python app.py", "Starting Flask application")
            return
    
    # Train ML model
    if not ml_model_exists or choice == 'y':
        run_command("python train_ml_model.py", "Training Machine Learning model")
    
    # Train DL model
    if not dl_model_exists or choice == 'y':
        choice = input("\nTrain Deep Learning model? This may take time (y/n): ").lower()
        if choice == 'y':
            run_command("python train_dl_model.py", "Training Deep Learning model")
        else:
            print("\n⚠ Skipping DL model training")
            print("Note: System will use ML model only")
    
    # Start application
    print("\n" + "="*60)
    print("ALL SETUP COMPLETE!")
    print("="*60)
    choice = input("\nStart Flask application now? (y/n): ").lower()
    
    if choice == 'y':
        run_command("python app.py", "Starting Flask application")
    else:
        print("\nTo start the application later, run:")
        print("  python app.py")
        print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        sys.exit(1)