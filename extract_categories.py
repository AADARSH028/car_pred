
import pickle
import json
import numpy as np

def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return list(obj)

try:
    with open('LinearRegressionModel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    data = {}
    
    # Try to find OneHotEncoder
    # Usually Pipeline -> ColumnTransformer -> OneHotEncoder
    
    if hasattr(model, 'steps'):
        for step_name, step in model.steps:
            # Check for OneHotEncoder directly
            if hasattr(step, 'categories_'):
                data[step_name] = [convert_to_list(c) for c in step.categories_]
            
            # Check for ColumnTransformer
            if hasattr(step, 'transformers_'):
                for t_name, t_transform, t_cols in step.transformers_:
                    if hasattr(t_transform, 'categories_'):
                        # t_transform.categories_ is a list of arrays
                        # match with t_cols
                        for i, col in enumerate(t_cols):
                            data[col] = convert_to_list(t_transform.categories_[i])
    
    print("JSON_START")
    print(json.dumps(data))
    print("JSON_END")

except Exception as e:
    print(e)
