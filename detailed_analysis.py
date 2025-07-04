#!/usr/bin/env python3
"""
Detailed analysis of differences between full_trivariate_M1.pkl and full_trivariate_M2.pkl
"""

import pickle
import numpy as np

def detailed_analysis():
    print('Loading files...')
    with open('outputs/pickles/full_trivariate_M1.pkl', 'rb') as f:
        data_m1 = pickle.load(f)

    with open('outputs/pickles/full_trivariate_M2.pkl', 'rb') as f:
        data_m2 = pickle.load(f)

    print('=== DETAILED ANALYSIS ===')
    
    for key in ['level_1', 'level_2', 'log_likelihood']:
        print(f'\n--- Key: {key} ---')
        val1, val2 = data_m1[key], data_m2[key]
        
        print(f'M1 type: {type(val1)}')
        print(f'M2 type: {type(val2)}')
        
        if hasattr(val1, 'shape'):
            print(f'M1 shape: {val1.shape}')
        if hasattr(val2, 'shape'):
            print(f'M2 shape: {val2.shape}')
        
        if hasattr(val1, '__len__') and not isinstance(val1, str):
            print(f'M1 length: {len(val1)}')
        if hasattr(val2, '__len__') and not isinstance(val2, str):
            print(f'M2 length: {len(val2)}')
        
        # Try numpy comparison for arrays
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            are_equal = np.array_equal(val1, val2)
            print(f'Arrays equal: {are_equal}')
            
            if not are_equal and val1.shape == val2.shape:
                diff = np.abs(val1 - val2)
                print(f'Max difference: {np.max(diff)}')
                print(f'Mean difference: {np.mean(diff)}')
                print(f'Std of differences: {np.std(diff)}')
                
                # Show some example differences
                non_zero_diffs = diff[diff > 1e-10]
                if len(non_zero_diffs) > 0:
                    print(f'Number of non-zero differences: {len(non_zero_diffs)}')
                    print(f'First few differences: {non_zero_diffs[:5]}')
        
        # For log_likelihood (scalar values)
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = abs(val1 - val2)
            print(f'Difference: {diff}')
            print(f'Relative difference: {diff / abs(val1) * 100:.6f}%')
        
        # For other types, try to get more info
        else:
            print(f'M1 value preview: {str(val1)[:100]}...' if len(str(val1)) > 100 else f'M1 value: {val1}')
            print(f'M2 value preview: {str(val2)[:100]}...' if len(str(val2)) > 100 else f'M2 value: {val2}')
            
            # If they're both lists or similar structures
            if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                if len(val1) == len(val2):
                    # Check element-wise differences
                    differences = []
                    for i, (v1, v2) in enumerate(zip(val1, val2)):
                        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                            if not np.array_equal(v1, v2):
                                differences.append(i)
                        else:
                            if v1 != v2:
                                differences.append(i)
                    
                    if differences:
                        print(f'Different elements at indices: {differences[:10]}...' if len(differences) > 10 else f'Different elements at indices: {differences}')
                    else:
                        print('All elements are identical')

if __name__ == "__main__":
    detailed_analysis()
