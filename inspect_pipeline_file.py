
import pickle
import sys

with open('pipeline_info.txt', 'w') as out:
    try:
        with open('LinearRegressionModel.pkl', 'rb') as f:
            model = pickle.load(f)
        
        out.write("Model Loaded\n")
        
        if hasattr(model, 'steps'):
            out.write(f"Pipeline Steps: {[s[0] for s in model.steps]}\n")
            
            for name, step in model.steps:
                if hasattr(step, 'transformers_'):
                    out.write(f"Found Transformer Step: {name}\n")
                    for t_name, t_transform, t_cols in step.transformers_:
                        out.write(f"  Transformer: {t_name}, Columns: {t_cols}\n")
                        if hasattr(t_transform, 'categories_'):
                            out.write(f"    Has categories. Count: {len(t_transform.categories_)}\n")
                            for i, cats in enumerate(t_transform.categories_):
                                out.write(f"    Category {i} (Length {len(cats)}): {list(cats)[:3]}...\n")

        if hasattr(model, 'feature_names_in_'):
            out.write(f"Feature Names In: {list(model.feature_names_in_)}\n")

    except Exception as e:
        out.write(f"Error: {e}\n")
