
import pickle
import pandas as pd
import numpy as np
import sys

try:
    with open('LinearRegressionModel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model)}")
    print(f"Model object: {model}")

    if hasattr(model, 'feature_names_in_'):
        print("Feature names:")
        for name in model.feature_names_in_:
            print(f" - {name}")
            
    # Check if it's a pipeline and has categories
    if hasattr(model, 'steps'): # It's likely a sklearn Pipeline
        print("\nPipeline Steps:")
        for name, step in model.steps:
            print(f" - {name}: {type(step)}")
            if hasattr(step, 'categories_'):
                print(f"   Categories: {step.categories_}")
            if hasattr(step, 'transformers_'): # ColumnTransformer
                for t_name, t_transform, t_cols in step.transformers_:
                    print(f"   Transformer {t_name}: {type(t_transform)} on {t_cols}")
                    if hasattr(t_transform, 'categories_'):
                        print(f"     Categories: {t_transform.categories_}")

except Exception as e:
    print(f"Error: {e}")
