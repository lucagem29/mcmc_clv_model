# mcmc_clv_model Project Setup Requirements

## Overview
This project uses Git LFS (Large File Storage) to manage large pickle files containing MCMC results. Follow these steps to properly set up your environment.

## Required Software

### 1. Git LFS (Large File Storage)
**Critical**: This project requires Git LFS to download the actual pickle files.

#### Installation:

**macOS:**
```bash
# Using Homebrew (recommended)
brew install git-lfs

# Using MacPorts
sudo port install git-lfs
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git-lfs
```

**Windows:**
- Download from: https://git-lfs.github.io/
- Or use: `winget install Git.GitLFS`

#### Initialize Git LFS:
```bash
git lfs install
```

### 2. Python Environment Setup

#### Create Virtual Environment:
```bash
# Create virtual environment
python3 -m venv prob_mcmc

# Activate virtual environment
# On macOS/Linux:
source prob_mcmc/bin/activate
# On Windows:
# prob_mcmc\Scripts\activate
```

#### Install Required Python Packages:
```bash
# With virtual environment activated
pip install numpy git-lfs pickle5
```

## Project Setup Steps

### 1. Clone Repository and Download LFS Files:
```bash
# Clone the repository
git clone [repository-url] mcmc_clv_model
cd mcmc_clv_model

# Download all LFS files
git lfs pull
```

### 2. Verify Setup:
Check that pickle files are properly downloaded:
```bash
ls -la outputs/pickles/
```

The files should be large (>10MB), not small text pointer files.

### 3. Test Pickle Loading:
```python
import pickle
import os

pickles_dir = 'outputs/pickles/'

# Test loading a pickle file
with open(os.path.join(pickles_dir, 'bivariate_M1.pkl'), 'rb') as f:
    draws_m1 = pickle.load(f)
    
print(f"Successfully loaded: {type(draws_m1)}")
```

## Troubleshooting

### Issue: UnpicklingError: invalid load key
**Symptom**: Error when loading pickle files
**Cause**: Git LFS files not downloaded (you have pointer files instead of actual data)
**Solution**: 
```bash
cd mcmc_clv_model
git lfs pull
```

### Issue: Small file sizes (< 1KB)
**Symptom**: Pickle files are only 133 bytes or similar
**Cause**: Git LFS not properly set up
**Solution**:
1. Install Git LFS: `brew install git-lfs` (macOS)
2. Initialize: `git lfs install`
3. Pull files: `git lfs pull`

### Issue: Git LFS not found
**Symptom**: `git lfs` command not recognized
**Solution**: Install Git LFS using the installation methods above

## File Structure
```
mcmc_clv_model/
├── outputs/
│   └── pickles/
│       ├── bivariate_M1.pkl      (~12MB)
│       ├── bivariate_M2.pkl      (~12MB)
│       ├── trivariate_M1.pkl     (~18MB)
│       ├── trivariate_M2.pkl     (~18MB)
│       └── other pickle files...
├── SETUP_REQUIREMENTS.md         (this file)
└── other project files...
```

## Important Notes

1. **Always activate your virtual environment** before running Python code
2. **Git LFS is mandatory** - the project will not work without it
3. **Check file sizes** after cloning to ensure LFS files downloaded properly
4. **Internet connection required** for initial LFS file download

## Quick Setup Script (macOS/Linux)
```bash
#!/bin/bash
# Quick setup script

# Install Git LFS (macOS with Homebrew)
brew install git-lfs

# Initialize Git LFS
git lfs install

# Create and activate virtual environment
python3 -m venv prob_mcmc
source prob_mcmc/bin/activate

# Install Python packages
pip install numpy git-lfs

# Clone and setup project (replace with actual repo URL)
# git clone [your-repo-url] mcmc_clv_model
# cd mcmc_clv_model
# git lfs pull

echo "Setup complete! Remember to activate your virtual environment before using the project."
```

## Contact
If you encounter issues, check that:
1. Git LFS is installed and initialized
2. Virtual environment is activated
3. All LFS files have been downloaded (`git lfs pull`)
4. File sizes are correct (see File Structure section above)
