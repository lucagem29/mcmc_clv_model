#!/usr/bin/env python3
"""
Script to compare full_trivariate_M1.pkl and full_trivariate_M2.pkl
"""

import pickle
import numpy as np
import sys

def compare_pickle_files():
    print('Loading full_trivariate_M1.pkl...')
    with open('outputs/pickles/full_trivariate_M1.pkl', 'rb') as f:
        data_m1 = pickle.load(f)

    print('Loading full_trivariate_M2.pkl...')
    with open('outputs/pickles/full_trivariate_M2.pkl', 'rb') as f:
        data_m2 = pickle.load(f)

    print(f'M1 type: {type(data_m1)}')
    print(f'M2 type: {type(data_m2)}')

    # Check if they're the same type
    if type(data_m1) != type(data_m2):
        print('❌ Files contain different data types')
        return False

    print('✅ Both files contain the same data type')

    if isinstance(data_m1, dict) and isinstance(data_m2, dict):
        print(f'M1 keys: {list(data_m1.keys())}')
        print(f'M2 keys: {list(data_m2.keys())}')
        
        # Compare keys
        if set(data_m1.keys()) != set(data_m2.keys()):
            print('❌ Different keys in dictionaries')
            return False
        
        print('✅ Same keys in both dictionaries')
        
        # Compare each key's content
        are_identical = True
        for key in data_m1.keys():
            print(f'\nComparing key: "{key}"')
            
            # Check if both values are arrays
            val1, val2 = data_m1[key], data_m2[key]
            
            if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                # Convert both to arrays for consistent comparison
                arr1 = np.asarray(val1)
                arr2 = np.asarray(val2)
                
                if not np.array_equal(arr1, arr2):
                    print(f'❌ Key "{key}": Arrays are different')
                    print(f'   M1 shape: {arr1.shape}, M2 shape: {arr2.shape}')
                    
                    # Show some statistics about the differences
                    if arr1.shape == arr2.shape:
                        diff = np.abs(arr1 - arr2)
                        print(f'   Max absolute difference: {np.max(diff)}')
                        print(f'   Mean absolute difference: {np.mean(diff)}')
                    
                    are_identical = False
                else:
                    print(f'✅ Key "{key}": Arrays are identical (shape: {arr1.shape})')
            else:
                # For non-array values
                try:
                    if val1 != val2:
                        print(f'❌ Key "{key}": Values are different')
                        print(f'   M1: {val1}')
                        print(f'   M2: {val2}')
                        are_identical = False
                    else:
                        print(f'✅ Key "{key}": Values are identical')
                except ValueError:
                    # Fallback for complex comparisons
                    print(f'⚠️  Key "{key}": Could not compare values directly')
                    are_identical = False
        
        if are_identical:
            print('\n🎉 CONCLUSION: Files are completely identical!')
            return True
        else:
            print('\n❌ CONCLUSION: Files are NOT identical!')
            return False
    
    else:
        # Handle non-dictionary data
        try:
            if isinstance(data_m1, np.ndarray) and isinstance(data_m2, np.ndarray):
                if np.array_equal(data_m1, data_m2):
                    print('🎉 CONCLUSION: Arrays are identical!')
                    return True
                else:
                    print('❌ CONCLUSION: Arrays are different!')
                    return False
            else:
                if data_m1 == data_m2:
                    print('🎉 CONCLUSION: Data structures are identical!')
                    return True
                else:
                    print('❌ CONCLUSION: Data structures are different!')
                    return False
        except Exception as e:
            print(f'Could not compare directly: {e}')
            return False

if __name__ == "__main__":
    compare_pickle_files()
