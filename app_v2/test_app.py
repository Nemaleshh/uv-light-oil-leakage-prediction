import sys
import os
import unittest

def run_all_tests():
    """Discover and run all tests in the tests/ directory."""
    print("Running UV Engine Oil Leak Detection Core Test Suite...")
    print("-" * 60)
    
    # Set the start directory to the 'tests' folder
    start_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    
    # Discover all tests matching 'test_*.py'
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    run_all_tests()
