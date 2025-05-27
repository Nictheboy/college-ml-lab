import json
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import time

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def calculate_msle(y_true, y_pred):
    """计算MSLE"""
    # 确保没有负值
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 1)  # 避免预测值为0
    return mean_squared_log_error(y_true, y_pred)

def main():
    """主函数"""
    print("=== 随机森林模型 ===\n")
    
    # 加载特征数据
    print("加载特征数据...")
    X_train = np.load(DATA_DIR / 'X_train.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    X_test = np.load(DATA_DIR / 'X_test.npy')
    
    # 加载元数据
    with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}")
    
    # 对目标值进行对数变换
    y_train_log = np.log1p(y_train)
    
    # 划分训练集和验证集
    X_tr, X_val, y_tr_log, y_val_log, y_tr, y_val = train_test_split(
        X_train, y_train_log, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集: {X_tr.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    # 训练随机森林模型
    print("\n训练随机森林模型...")
    start_time = time.time()
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # 在对数空间训练
    rf_model.fit(X_tr, y_tr_log)
    
    training_time = time.time() - start_time
    print(f"训练时间: {training_time:.2f} 秒")
    
    # 验证集预测
    print("\n验证集评估...")
    y_val_pred_log = rf_model.predict(X_val)
    y_val_pred = np.expm1(y_val_pred_log)  # 转换回原始空间
    
    val_msle = calculate_msle(y_val, y_val_pred)
    print(f"验证集MSLE: {val_msle:.6f}")
    print(f"验证集RMSLE: {np.sqrt(val_msle):.6f}")
    
    # 全量训练集评估
    print("\n全量训练集评估...")
    y_train_pred_log = rf_model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)
    
    train_msle = calculate_msle(y_train, y_train_pred)
    print(f"训练集MSLE: {train_msle:.6f}")
    print(f"训练集RMSLE: {np.sqrt(train_msle):.6f}")
    
    # 特征重要性分析
    print("\n特征重要性（前10个）:")
    feature_importance = rf_model.feature_importances_
    feature_names = metadata['feature_names']
    
    # 排序特征重要性
    indices = np.argsort(feature_importance)[::-1]
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # 生成测试集预测
    print("\n生成测试集预测...")
    y_test_pred_log = rf_model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # 保存预测结果
    predictions = []
    for query_id, pred in zip(metadata['test_ids'], y_test_pred):
        predictions.append({
            "query_id": query_id,
            "predicted_rows": int(max(1, pred))  # 确保至少为1
        })
    
    output_file = DATA_DIR / "predictions_rf.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n预测结果已保存到: {output_file}")
    
    # 保存模型
    model_file = DATA_DIR / "rf_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"模型已保存到: {model_file}")
    
    # 保存模型结果
    rf_results = {
        'model': 'random_forest',
        'train_msle': float(train_msle),
        'train_rmsle': float(np.sqrt(train_msle)),
        'val_msle': float(val_msle),
        'val_rmsle': float(np.sqrt(val_msle)),
        'training_time': training_time,
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt'
        }
    }
    
    with open(DATA_DIR / 'rf_results.json', 'w') as f:
        json.dump(rf_results, f, indent=2)
    
    print(f"\n随机森林模型评估完成！")
    print(f"验证集MSLE: {val_msle:.6f}")
    print(f"验证集RMSLE: {np.sqrt(val_msle):.6f}")

if __name__ == "__main__":
    main() 