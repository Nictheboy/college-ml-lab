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
    """计算MSLE"""
    # 确保没有负值
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 1)  # 避免预测值为0
    return mean_squared_log_error(y_true, y_pred)

def main():
    """主函数"""
    print("=== XGBoost模型 V2 (改进特征) ===\n")
    
    # 加载特征数据
    print("加载改进的特征数据...")
    X_train = np.load(DATA_DIR / 'X_train_v2.npy')
    y_train = np.load(DATA_DIR / 'y_train_v2.npy')
    X_test = np.load(DATA_DIR / 'X_test_v2.npy')
    
    # 加载元数据
    with open(DATA_DIR / 'metadata_v2.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}")
    print(f"特征数量增加: 36 -> {X_train.shape[1]} (+{X_train.shape[1]-36})")
    
    # 对目标值进行对数变换
    y_train_log = np.log1p(y_train)
    
    # 划分训练集和验证集
    X_tr, X_val, y_tr_log, y_val_log, y_tr, y_val = train_test_split(
        X_train, y_train_log, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\n训练集: {X_tr.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    # 创建DMatrix
    dtrain = xgb.DMatrix(X_tr, label=y_tr_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)
    
    # XGBoost参数（针对更多特征进行调整）
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,  # 增加深度以利用更多特征
        'learning_rate': 0.05,  # 降低学习率
        'subsample': 0.8,
        'colsample_bytree': 0.7,  # 降低列采样率
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.5,  # 增加正则化
        'reg_lambda': 2.0,
        'seed': 42,
        'n_jobs': -1
    }
    
    # 训练XGBoost模型
    print("\n训练XGBoost模型...")
    start_time = time.time()
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,  # 增加轮数
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    training_time = time.time() - start_time
    print(f"\n训练时间: {training_time:.2f} 秒")
    print(f"最佳迭代轮数: {xgb_model.best_iteration}")
    
    # 验证集预测
    print("\n验证集评估...")
    y_val_pred_log = xgb_model.predict(dval)
    y_val_pred = np.expm1(y_val_pred_log)  # 转换回原始空间
    
    val_msle = calculate_msle(y_val, y_val_pred)
    print(f"验证集MSLE: {val_msle:.6f}")
    print(f"验证集RMSLE: {np.sqrt(val_msle):.6f}")
    
    # 全量训练集评估
    print("\n全量训练集评估...")
    dtrain_full = xgb.DMatrix(X_train)
    y_train_pred_log = xgb_model.predict(dtrain_full)
    y_train_pred = np.expm1(y_train_pred_log)
    
    train_msle = calculate_msle(y_train, y_train_pred)
    print(f"训练集MSLE: {train_msle:.6f}")
    print(f"训练集RMSLE: {np.sqrt(train_msle):.6f}")
    
    # 特征重要性分析
    print("\n特征重要性（前20个）:")
    feature_importance = xgb_model.get_score(importance_type='gain')
    feature_names = metadata['feature_names']
    
    # 创建特征名映射
    feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
    
    # 排序特征重要性
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(sorted_features[:20]):
        feat_name = feature_map.get(feat, feat)
        print(f"  {i+1:2d}. {feat_name}: {score:.2f}")
    
    # 分析新特征的贡献
    print("\n新特征贡献分析:")
    new_feature_importance = []
    for feat, score in feature_importance.items():
        feat_idx = int(feat[1:])
        if feat_idx >= 36:  # 新特征
            feat_name = feature_names[feat_idx]
            new_feature_importance.append((feat_name, score))
    
    new_feature_importance.sort(key=lambda x: x[1], reverse=True)
    print(f"使用的新特征数: {len(new_feature_importance)}")
    print("最重要的新特征（前10个）:")
    for i, (feat_name, score) in enumerate(new_feature_importance[:10]):
        print(f"  {i+1:2d}. {feat_name}: {score:.2f}")
    
    # 生成测试集预测
    print("\n生成测试集预测...")
    dtest = xgb.DMatrix(X_test)
    y_test_pred_log = xgb_model.predict(dtest)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # 保存预测结果
    predictions = []
    for query_id, pred in zip(metadata['test_ids'], y_test_pred):
        predictions.append({
            "query_id": query_id,
            "predicted_rows": int(max(1, pred))  # 确保至少为1
        })
    
    output_file = DATA_DIR / "predictions_xgb_v2.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n预测结果已保存到: {output_file}")
    
    # 保存模型
    model_file = DATA_DIR / "xgb_model_v2.json"
    xgb_model.save_model(model_file)
    print(f"模型已保存到: {model_file}")
    
    # 保存模型结果
    xgb_v2_results = {
        'model': 'xgboost_v2',
        'train_msle': float(train_msle),
        'train_rmsle': float(np.sqrt(train_msle)),
        'val_msle': float(val_msle),
        'val_rmsle': float(np.sqrt(val_msle)),
        'training_time': training_time,
        'best_iteration': xgb_model.best_iteration,
        'num_features': X_train.shape[1],
        'hyperparameters': params
    }
    
    with open(DATA_DIR / 'xgb_v2_results.json', 'w') as f:
        json.dump(xgb_v2_results, f, indent=2)
    
    # 比较改进
    print("\n=== 性能改进对比 ===")
    # 加载之前的结果
    with open(DATA_DIR / 'xgb_results.json', 'r') as f:
        xgb_v1_results = json.load(f)
    
    print(f"XGBoost V1 (36特征): 验证集MSLE = {xgb_v1_results['val_msle']:.6f}")
    print(f"XGBoost V2 (78特征): 验证集MSLE = {val_msle:.6f}")
    
    improvement = (xgb_v1_results['val_msle'] - val_msle) / xgb_v1_results['val_msle'] * 100
    if improvement > 0:
        print(f"改进: {improvement:.2f}%")
    else:
        print(f"性能下降: {-improvement:.2f}%")
    
    print(f"\nXGBoost V2模型评估完成！")
    print(f"验证集MSLE: {val_msle:.6f}")
    print(f"验证集RMSLE: {np.sqrt(val_msle):.6f}")

if __name__ == "__main__":
    main() 