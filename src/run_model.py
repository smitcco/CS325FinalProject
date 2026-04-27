" Run model "

import matplotlib
import runpy
import os
matplotlib.use('Agg')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\n-- Step 1: Data Cleaning/Preprocessing -- ")
runpy.run_path('src/data_processing.py')

print("\n-- Step 2: Model Selection/Training -- ")
runpy.run_path('src/model_select_training.py')

print("\n-- Step 3: Feature Selection -- ")
runpy.run_path('src/feature_selection.py')

print("\n-- Step 4: Model Eval -- ")
runpy.run_path('src/model_evaluation.py')

print("\nRuns done. Results located in results/")
