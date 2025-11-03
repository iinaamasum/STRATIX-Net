"""
Simple test script to verify all imports work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test all imports"""
    print("Testing imports...")

    try:
        from Model import get_model, count_parameters

        print("✓ Model imports successful")
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False

    try:
        from Load_Dataset import get_data_loaders

        print("✓ Load_Dataset imports successful")
    except Exception as e:
        print(f"✗ Load_Dataset import failed: {e}")
        return False

    try:
        from Preprocessing import get_transforms

        print("✓ Preprocessing imports successful")
    except Exception as e:
        print(f"✗ Preprocessing import failed: {e}")
        return False

    try:
        from Training import train_and_evaluate, get_optimizer_scheduler

        print("✓ Training imports successful")
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        return False

    try:
        from Evaluation import test_model, plot_metrics, count_parameters

        print("✓ Evaluation imports successful")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        return False

    print("\nAll imports successful! ✓")
    return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
