import json
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_log_error

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train_transformed.jsonl"
TEST_FILE = DATA_DIR / "test_transformed.jsonl"

def calculate_msle(y_true, y_pred):
    """计算MSLE"""
    # 确保没有负值
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    
    # 手动计算MSLE以便更好地理解
    log_true = np.log1p(y_true)  # log(1 + y)
    log_pred = np.log1p(y_pred)
    squared_log_error = (log_true - log_pred) ** 2
    msle = np.mean(squared_log_error)
    
    # 验证与sklearn的结果一致
    sklearn_msle = mean_squared_log_error(y_true, y_pred)
    assert np.isclose(msle, sklearn_msle), f"Manual MSLE {msle} != sklearn MSLE {sklearn_msle}"
    
    return msle

def extract_plan_rows(file_path: Path):
    """从文件中提取执行计划的预估行数"""
    plan_rows = []
    actual_rows = []
    query_ids = []
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # 提取执行计划的预估行数
            plan_features = item['features'].get('plan_features', {})
            pred_rows = plan_features.get('plan_rows', 1)  # 默认为1避免0
            plan_rows.append(pred_rows)
            
            # 提取实际行数（如果存在）
            if 'target_actual_rows' in item:
                actual_rows.append(item['target_actual_rows'])
            
            query_ids.append(item['query_id'])
    
    return np.array(plan_rows), np.array(actual_rows) if actual_rows else None, query_ids

def main():
    """主函数"""
    print("=== 基线模型：使用执行计划预估行数 ===\n")
    
    # 提取训练集数据
    print("加载训练集数据...")
    train_pred, train_actual, train_ids = extract_plan_rows(TRAIN_FILE)
    print(f"训练集样本数: {len(train_pred)}")
    
    # 计算训练集MSLE
    train_msle = calculate_msle(train_actual, train_pred)
    print(f"\n训练集MSLE: {train_msle:.6f}")
    print(f"训练集RMSLE: {np.sqrt(train_msle):.6f}")
    
    # 分析预测误差
    print("\n预测误差分析:")
    log_errors = np.log1p(train_pred) - np.log1p(train_actual)
    print(f"  平均对数误差: {np.mean(log_errors):.4f}")
    print(f"  对数误差标准差: {np.std(log_errors):.4f}")
    
    # 计算相对误差
    relative_errors = np.abs(train_pred - train_actual) / (train_actual + 1)
    print(f"  平均相对误差: {np.mean(relative_errors):.2%}")
    print(f"  中位数相对误差: {np.median(relative_errors):.2%}")
    
    # 分析预测偏差
    overestimate_ratio = np.sum(train_pred > train_actual) / len(train_pred)
    print(f"\n预测偏差:")
    print(f"  高估比例: {overestimate_ratio:.2%}")
    print(f"  低估比例: {1 - overestimate_ratio:.2%}")
    
    # 按误差大小分析
    large_error_mask = np.abs(log_errors) > 1  # 对数误差大于1
    print(f"  大误差样本比例: {np.sum(large_error_mask) / len(train_pred):.2%}")
    
    # 生成测试集预测
    print("\n生成测试集预测...")
    test_pred, _, test_ids = extract_plan_rows(TEST_FILE)
    print(f"测试集样本数: {len(test_pred)}")
    
    # 保存预测结果
    predictions = []
    for query_id, pred in zip(test_ids, test_pred):
        predictions.append({
            "query_id": query_id,
            "predicted_rows": int(pred)  # 确保是整数
        })
    
    output_file = DATA_DIR / "predictions_baseline.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n预测结果已保存到: {output_file}")
    
    # 显示一些预测示例
    print("\n预测示例（前5个）:")
    for i in range(min(5, len(predictions))):
        print(f"  Query {predictions[i]['query_id']}: {predictions[i]['predicted_rows']:,} rows")
    
    # 保存基线模型性能
    baseline_results = {
        'model': 'baseline_plan_rows',
        'train_msle': float(train_msle),
        'train_rmsle': float(np.sqrt(train_msle)),
        'feature_description': '使用执行计划的预估行数作为预测'
    }
    
    with open(DATA_DIR / 'baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"\n基线模型评估完成！")
    print(f"MSLE: {train_msle:.6f}")
    print(f"RMSLE: {np.sqrt(train_msle):.6f}")

if __name__ == "__main__":
    main() 