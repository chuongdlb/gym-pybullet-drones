"""
Master test runner for all GateAviary modifications.
Runs all test suites and generates a summary report.
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_file):
    """Run a single test file and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0

def main():
    test_dir = Path(__file__).parent
    
    tests = [
        test_dir / "test_gate_configuration.py",
        test_dir / "test_reward_function.py",
        test_dir / "test_observation_space.py",
        test_dir / "test_integration.py",
    ]
    
    print("\n" + "="*70)
    print("GATEAVIARY TEST SUITE")
    print("="*70)
    print(f"Testing modifications to GateAviary environment:")
    print(f"  - 6 gates (changed from 5)")
    print(f"  - Track randomization per episode")
    print(f"  - Racing-optimized reward function")
    print(f"  - Enhanced observation space (next 2 gates + velocity + time)")
    print("="*70)
    
    results = {}
    
    for test_file in tests:
        if test_file.exists():
            test_name = test_file.stem
            success = run_test(test_file)
            results[test_name] = success
        else:
            print(f"⚠ Test file not found: {test_file}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nTotal: {passed}/{total} test suites passed")
    
    if all(results.values()):
        print("\n✓ ALL TESTS PASSED - Environment modifications verified!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Review output above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
