
import pickle
import pandas as pd
import numpy as np

try:
    with open('LinearRegressionModel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Model Loaded")
    
    if hasattr(model, 'steps'):
        print("Pipeline Steps:", [s[0] for s in model.steps])
        
        for name, step in model.steps:
            if name == 'column_transformer' or hasattr(step, 'transformers_'):
                print(f"Found Transformer Step: {name}")
                for t_name, t_transform, t_cols in step.transformers_:
                    print(f"  Transformer: {t_name}, Columns: {t_cols}")
                    if hasattr(t_transform, 'categories_'):
                        print(f"    Has categories. Count: {len(t_transform.categories_)}")
                        for i, cats in enumerate(t_transform.categories_):
                            print(f"    Category {i} (Length {len(cats)}): {cats[:5]}...")
                            # Output logic to help us map
                            if len(cats) < 10:
                                print(f"    -> Likely fuel_type")
                            elif len(cats) < 50:
                                print(f"    -> Likely company")
                            else:
                                print(f"    -> Likely name/model")
                                
    # Also check what features the model expects
    if hasattr(model, 'feature_names_in_'):
        print("Feature Names In:", model.feature_names_in_)

except Exception as e:
    print(f"Error: {e}")
