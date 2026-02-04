
import pickle
import sys

try:
    with open('LinearRegressionModel.pkl', 'rb') as f:
        model = pickle.load(f)
    
    t = str(type(model))
    print("TYPE_START")
    print(t)
    print("TYPE_END")
    
    if hasattr(model, 'feature_names_in_'):
        print("FEATURES_START")
        print(list(model.feature_names_in_))
        print("FEATURES_END")

except Exception as e:
    print(e)
