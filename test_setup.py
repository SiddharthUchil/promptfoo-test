#!/usr/bin/env python3
"""
Test setup script to verify all dependencies and configurations.
Run this before running the main evaluation to catch issues early.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_python_dependencies():
    """Test if all Python dependencies are available."""
    print("Testing Python dependencies...")
    
    required_packages = [
        'pandas', 'openpyxl', 'scikit-learn', 'yaml', 'openai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'openpyxl':
                import openpyxl
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("All Python dependencies available!")
    return True

def test_node_dependencies():
    """Test if Node.js and promptfoo are available."""
    print("\nTesting Node.js dependencies...")
    
    try:
        # Test Node.js
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            print(f"  ✓ Node.js {result.stdout.strip()}")
        else:
            print("  ✗ Node.js not found")
            return False
            
        # Test promptfoo
        result = subprocess.run(['npx', 'promptfoo', '--version'], 
                              capture_output=True, text=True,
                              encoding='utf-8', errors='ignore', shell=True)
        if result.returncode == 0:
            print(f"  ✓ promptfoo available")
        else:
            print("  ✗ promptfoo not available")
            print("Install with: npm install")
            return False
            
    except Exception as e:
        print(f"  ✗ Error testing Node.js dependencies: {e}")
        return False
    
    print("Node.js dependencies available!")
    return True

def test_required_files():
    """Test if all required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        'promptfooconfig.yaml',
        'evaluation_data.xlsx', 
        'prompt_v1.md',
        'prompt_v2.md'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            missing_files.append(file)
            print(f"  ✗ {file} (missing)")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    print("All required files found!")
    return True

def test_environment_variables():
    """Test if required environment variables are set."""
    print("\nTesting environment variables...")
    
    if os.getenv('OPENAI_API_KEY'):
        print("  ✓ OPENAI_API_KEY is set")
        return True
    else:
        print("  ✗ OPENAI_API_KEY not set")
        print("Set with: export OPENAI_API_KEY=your_key_here")
        print("Or create a .env file with: OPENAI_API_KEY=your_key_here")
        return False

def main():
    """Run all tests."""
    print("Promptfoo RAG Evaluation Setup Test")
    print("=" * 50)
    
    tests = [
        test_python_dependencies,
        test_node_dependencies, 
        test_required_files,
        test_environment_variables
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Ready to run evaluation.")
        print("Run: python evaluate.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 