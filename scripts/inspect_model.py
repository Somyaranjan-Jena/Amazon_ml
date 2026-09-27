import sys
sys.path.insert(0, ".")
import joblib
import xgboost as xgb

model = joblib.load("cache/lgb_model.joblib")
print(f"Model type: {type(model).__name__}")

has_booster = hasattr(model, "booster")
print(f"Has booster attr: {has_booster}")

if has_booster:
    bst = model.booster
    print(f"Booster type: {type(bst).__name__}")
    print(f"Best iteration: {bst.best_iteration}")
    print(f"Num features: {bst.num_features()}")
    print(f"Feature names: {bst.feature_names}")
    imp = bst.get_score(importance_type="gain")
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    print("Top features by gain:")
    for name, gain in sorted_imp[:15]:
        print(f"  {name}: {gain:.1f}")
elif hasattr(model, "predict_proba"):
    print("Model has predict_proba method (sklearn-compatible)")
