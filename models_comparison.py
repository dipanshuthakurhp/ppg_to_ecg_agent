import os
import numpy as np
import pandas as pd
from src.models_zoo import Models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run_training(model_enum):
    clear_segments_path = "clear_segments.npy"
    clear_segments = np.load(clear_segments_path, allow_pickle=True)
    clear_segments = np.array(clear_segments.tolist())

    X = clear_segments[:, 0, :]
    y = clear_segments[:, 1, :]

    X = (((X - np.min(X)) / (np.max(X) - np.min(X))) * 2)- 1
    y = (((y - np.min(y))/ (np.max(y) - np.min(y))) * 2) - 1

    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model_name = model_enum.name
    model = model_enum.value
    

    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 20, batch_size = 32)

    y_pred = model.predict(X_val, verbose = 2)

    y_true_flat = y_val.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mse = mean_squared_error(y_true=y_true_flat, y_pred=y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=y_true_flat, y_pred=y_pred_flat)
    r2 = r2_score(y_true=y_true_flat, y_pred=y_pred_flat)

    model_path = os.path.join(f'models/saved_keras/{model_name}.h5')
    model.save(model_path)

    result = {
        "model_name": model_name,
        "num_params": model.count_params(),
        "rmse": float(rmse),
        "mse": float(mse),
        "mae": float(mae),
        "r2_score": float(r2)
    }

    return result


results = []

print("Models content:", list(Models))
for model_enum in Models:
    print(f"==================Running for model {model_enum.name}==============")
    result = run_training(model_enum)
    results.append(result)

results_df = pd.DataFrame(results)
results_df.to_csv("reports/model_metrics.csv", index=False)
