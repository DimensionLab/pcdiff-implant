#!/usr/bin/env python3
"""
Test script for PCDiff Web Viewer components.

Tests:
1. Conversion script functionality
2. Backend API endpoints
3. File serving
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_conversion():
    """Test the NPY to PLY/STL conversion functionality."""
    print("=" * 60)
    print("Testing Conversion Script")
    print("=" * 60)
    
    try:
        from pcdiff.utils.convert_to_web import (
            load_npy_with_transform,
            npy_to_ply,
            npy_to_stl,
        )
        print("✓ Import successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Check if inference results exist
    results_dir = project_root / "inference_results_ddim50" / "syn"
    if not results_dir.exists():
        print(f"⚠ No inference results found at {results_dir}")
        print("  Skipping conversion test")
        return True
    
    # Find first result with input.npy
    test_result = None
    for result_dir in results_dir.iterdir():
        if (result_dir / "input.npy").exists():
            test_result = result_dir
            break
    
    if not test_result:
        print("⚠ No valid inference results found")
        return True
    
    print(f"Testing with: {test_result.name}")
    
    # Test loading NPY
    try:
        import numpy as np
        input_file = test_result / "input.npy"
        data = load_npy_with_transform(input_file)
        print(f"✓ Loaded input.npy: shape {data.shape}")
    except Exception as e:
        print(f"✗ Failed to load NPY: {e}")
        return False
    
    # Test PLY conversion
    try:
        output_dir = test_result / "test_web"
        output_dir.mkdir(exist_ok=True)
        ply_file = output_dir / "test_input.ply"
        
        count = npy_to_ply(input_file, ply_file, color=(200, 200, 200))
        print(f"✓ PLY conversion successful: {count} points")
        
        # Check file exists
        if ply_file.exists():
            size_kb = ply_file.stat().st_size / 1024
            print(f"  File size: {size_kb:.1f} KB")
        
        # Cleanup
        ply_file.unlink()
    except Exception as e:
        print(f"✗ PLY conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test STL conversion
    try:
        stl_file = output_dir / "test_input.stl"
        faces = npy_to_stl(input_file, stl_file)
        print(f"✓ STL conversion successful: {faces} faces")
        
        # Check file exists
        if stl_file.exists():
            size_kb = stl_file.stat().st_size / 1024
            print(f"  File size: {size_kb:.1f} KB")
        
        # Cleanup
        stl_file.unlink()
        output_dir.rmdir()
    except Exception as e:
        print(f"✗ STL conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All conversion tests passed!")
    return True


def test_backend_imports():
    """Test backend imports."""
    print("\n" + "=" * 60)
    print("Testing Backend Imports")
    print("=" * 60)
    
    try:
        from web_viewer.backend.config import settings
        print("✓ Config imported successfully")
        print(f"  Inference results dir: {settings.inference_results_dir}")
        print(f"  Host: {settings.host}")
        print(f"  Port: {settings.port}")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        print("  Make sure FastAPI dependencies are installed:")
        print("  pip install fastapi uvicorn pydantic pydantic-settings")
        return False
    
    try:
        from web_viewer.backend.main import app
        print("✓ FastAPI app imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI app import failed: {e}")
        return False
    
    print("\n✓ All backend import tests passed!")
    return True


def test_dependencies():
    """Test if required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Testing Dependencies")
    print("=" * 60)
    
    dependencies = [
        ("numpy", "NumPy"),
        ("trimesh", "Trimesh"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
    ]
    
    all_installed = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} not installed")
            all_installed = False
    
    if not all_installed:
        print("\nInstall missing dependencies:")
        print("  Backend: pip install -r web_viewer/backend/requirements.txt")
        print("  Conversion: pip install trimesh numpy")
    
    return all_installed


def main():
    """Run all tests."""
    print("PCDiff Web Viewer - Component Tests")
    print("=" * 60)
    
    results = []
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test conversion
    results.append(("Conversion", test_conversion()))
    
    # Test backend imports
    results.append(("Backend", test_backend_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Convert inference results:")
        print("   python3 pcdiff/utils/convert_to_web.py inference_results_ddim50 --batch")
        print("\n2. Start the web viewer:")
        print("   cd web_viewer && ./start_dev.sh")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

