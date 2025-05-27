import json
import numpy as np
from pathlib import Path
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import time

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def calculate_msle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 1)
    return mean_squared_log_error(y_true, y_pred)

def main():
    print("=== XGBoost模型 V3 (更鲁棒特征) ===\n")
    
    print("加载特征数据 V3...")
    X_train = np.load(DATA_DIR / 'X_train_v3.npy')
    y_train = np.load(DATA_DIR / 'y_train_v3.npy')
    X_test = np.load(DATA_DIR / 'X_test_v3.npy')
    
    with open(DATA_DIR / 'metadata_v3.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}")
    print(f"特征数量: {X_train.shape[1]}")
    
    y_train_log = np.log1p(y_train)
    
    X_tr, X_val, y_tr_log, y_val_log, _, y_val = train_test_split(
        X_train, y_train_log, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集: {X_tr.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10, 
        'learning_rate': 0.03, # Slightly lower LR for more rounds
        'subsample': 0.8,
        'colsample_bytree': 0.6, # Sample fewer features per tree
        'min_child_weight': 3,
        'gamma': 0.15,
        'reg_alpha': 0.6,
        'reg_lambda': 2.2,
        'seed': 42,
        'n_jobs': -1
    }
    
    print("\n训练XGBoost模型 V3...")
    start_time = time.time()
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=700, # More rounds
        evals=evals,
        early_stopping_rounds=50, # Increased patience
        verbose_eval=50
    )
    
    training_time = time.time() - start_time
    print(f"\n训练时间: {training_time:.2f} 秒")
    print(f"最佳迭代轮数: {xgb_model.best_iteration}")
    
    print("\n验证集评估...")
    y_val_pred_log = xgb_model.predict(dval)
    y_val_pred = np.expm1(y_val_pred_log)
    val_msle = calculate_msle(y_val, y_val_pred)
    print(f"验证集MSLE (V3): {val_msle:.6f}")
    print(f"验证集RMSLE (V3): {np.sqrt(val_msle):.6f}")
    
    print("\n全量训练集评估...")
    dtrain_full = xgb.DMatrix(X_train)
    y_train_pred_log = xgb_model.predict(dtrain_full)
    y_train_pred = np.expm1(y_train_pred_log)
    train_msle = calculate_msle(y_train, y_train_pred)
    print(f"训练集MSLE (V3): {train_msle:.6f}")
    print(f"训练集RMSLE (V3): {np.sqrt(train_msle):.6f}")
    
    print("\n生成测试集预测 (V3)...")
    dtest = xgb.DMatrix(X_test)
    y_test_pred_log = xgb_model.predict(dtest)
    y_test_pred = np.expm1(y_test_pred_log)
    
    predictions = []
    for query_id, pred in zip(metadata['test_ids'], y_test_pred):
        predictions.append({"query_id": query_id, "predicted_rows": int(max(1, pred))})
    
    output_file = DATA_DIR / "predictions_xgb_v3.json"
    with open(output_file, 'w') as f: json.dump(predictions, f, indent=2)
    print(f"\n预测结果已保存到: {output_file}")
    
    model_file = DATA_DIR / "xgb_model_v3.json"
    xgb_model.save_model(model_file)
    print(f"模型已保存到: {model_file}")
    
    xgb_v3_results = {
        'model': 'xgboost_v3',
        'train_msle': float(train_msle),
        'train_rmsle': float(np.sqrt(train_msle)),
        'val_msle': float(val_msle),
        'val_rmsle': float(np.sqrt(val_msle)),
        'training_time': training_time,
        'best_iteration': xgb_model.best_iteration,
        'num_features': X_train.shape[1],
        'hyperparameters': params
    }
    with open(DATA_DIR / 'xgb_v3_results.json', 'w') as f: json.dump(xgb_v3_results, f, indent=2)
    
    print("\n=== 性能改进对比 V2 vs V3 ===")
    with open(DATA_DIR / 'xgb_v2_results.json', 'r') as f: xgb_v2_results = json.load(f)
    print(f"XGBoost V2 ({xgb_v2_results['num_features']}特征): 验证集MSLE = {xgb_v2_results['val_msle']:.6f}")
    print(f"XGBoost V3 ({X_train.shape[1]}特征): 验证集MSLE = {val_msle:.6f}")
    improvement = (xgb_v2_results['val_msle'] - val_msle) / xgb_v2_results['val_msle'] * 100
    if improvement > 0: print(f"改进: {improvement:.2f}%")
    else: print(f"性能下降: {-improvement:.2f}%")
    
    print(f"\nXGBoost V3模型评估完成！")
    print(f"验证集MSLE: {val_msle:.6f}")

if __name__ == "__main__":
    main() 