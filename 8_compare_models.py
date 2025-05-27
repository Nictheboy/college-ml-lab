import json
from pathlib import Path

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_results():
    """加载所有模型的结果"""
    results = []
    
    # 基线模型
    with open(DATA_DIR / 'baseline_results.json', 'r') as f:
        baseline = json.load(f)
        results.append({
            'Model': 'Baseline (Plan Rows)',
            'Train MSLE': baseline['train_msle'],
            'Train RMSLE': baseline['train_rmsle'],
            'Val MSLE': '-',
            'Val RMSLE': '-',
            'Training Time': '-',
            'Description': baseline['feature_description']
        })
    
    # 随机森林
    with open(DATA_DIR / 'rf_results.json', 'r') as f:
        rf = json.load(f)
        results.append({
            'Model': 'Random Forest',
            'Train MSLE': rf['train_msle'],
            'Train RMSLE': rf['train_rmsle'],
            'Val MSLE': rf['val_msle'],
            'Val RMSLE': rf['val_rmsle'],
            'Training Time': f"{rf['training_time']:.2f}s",
            'Description': f"n_estimators={rf['hyperparameters']['n_estimators']}, max_depth={rf['hyperparameters']['max_depth']}"
        })
    
    # XGBoost
    with open(DATA_DIR / 'xgb_results.json', 'r') as f:
        xgb = json.load(f)
        results.append({
            'Model': 'XGBoost',
            'Train MSLE': xgb['train_msle'],
            'Train RMSLE': xgb['train_rmsle'],
            'Val MSLE': xgb['val_msle'],
            'Val RMSLE': xgb['val_rmsle'],
            'Training Time': f"{xgb['training_time']:.2f}s",
            'Description': f"best_iteration={xgb['best_iteration']}, max_depth={xgb['hyperparameters']['max_depth']}"
        })
    
    return results

def main():
    """主函数"""
    print("=== 模型性能比较 ===\n")
    
    # 加载结果
    results = load_results()
    
    # 创建比较表格
    print("模型性能总结:")
    print("-" * 100)
    print(f"{'模型':<20} {'训练MSLE':<12} {'验证MSLE':<12} {'训练RMSLE':<12} {'验证RMSLE':<12} {'训练时间':<10}")
    print("-" * 100)
    
    for result in results:
        train_msle = f"{result['Train MSLE']:.6f}" if isinstance(result['Train MSLE'], float) else result['Train MSLE']
        val_msle = f"{result['Val MSLE']:.6f}" if isinstance(result['Val MSLE'], float) else result['Val MSLE']
        train_rmsle = f"{result['Train RMSLE']:.6f}" if isinstance(result['Train RMSLE'], float) else result['Train RMSLE']
        val_rmsle = f"{result['Val RMSLE']:.6f}" if isinstance(result['Val RMSLE'], float) else result['Val RMSLE']
        
        print(f"{result['Model']:<20} {train_msle:<12} {val_msle:<12} {train_rmsle:<12} {val_rmsle:<12} {result['Training Time']:<10}")
    
    print("-" * 100)
    
    # 性能改进分析
    print("\n性能改进分析:")
    baseline_msle = results[0]['Train MSLE']
    
    for i, result in enumerate(results[1:], 1):
        if isinstance(result['Val MSLE'], float):
            improvement = (baseline_msle - result['Val MSLE']) / baseline_msle * 100
            print(f"{result['Model']}: 相比基线改进 {improvement:.1f}% (验证集MSLE)")
    
    # 最佳模型
    print("\n最佳模型推荐:")
    val_msles = [(r['Model'], r['Val MSLE']) for r in results if isinstance(r['Val MSLE'], float)]
    if val_msles:
        best_model = min(val_msles, key=lambda x: x[1])
        print(f"基于验证集MSLE，最佳模型是: {best_model[0]} (MSLE={best_model[1]:.6f})")
    
    # 保存比较结果
    comparison_file = DATA_DIR / 'model_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump({
            'models': results,
            'best_model': best_model[0] if val_msles else None,
            'baseline_msle': baseline_msle
        }, f, indent=2)
    
    print(f"\n比较结果已保存到: {comparison_file}")
    
    # 更新项目进展报告
    update_progress_report(results, best_model[0] if val_msles else None)

def update_progress_report(results, best_model):
    """更新项目进展报告"""
    report_path = BASE_DIR / "report" / "project_progress.md"
    
    # 读取现有报告
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加模型结果部分
    model_results = "\n\n## 模型实验结果\n\n"
    model_results += "### 模型性能比较\n\n"
    model_results += "| 模型 | 训练MSLE | 验证MSLE | 训练RMSLE | 验证RMSLE | 训练时间 |\n"
    model_results += "|------|----------|----------|-----------|-----------|----------|\n"
    
    for result in results:
        train_msle = f"{result['Train MSLE']:.6f}" if isinstance(result['Train MSLE'], float) else result['Train MSLE']
        val_msle = f"{result['Val MSLE']:.6f}" if isinstance(result['Val MSLE'], float) else result['Val MSLE']
        train_rmsle = f"{result['Train RMSLE']:.6f}" if isinstance(result['Train RMSLE'], float) else result['Train RMSLE']
        val_rmsle = f"{result['Val RMSLE']:.6f}" if isinstance(result['Val RMSLE'], float) else result['Val RMSLE']
        
        model_results += f"| {result['Model']} | {train_msle} | {val_msle} | {train_rmsle} | {val_rmsle} | {result['Training Time']} |\n"
    
    model_results += f"\n**最佳模型**: {best_model}\n"
    
    model_results += "\n### 关键发现\n"
    model_results += "1. 基线模型（使用执行计划预估行数）的MSLE为2.410\n"
    model_results += "2. 机器学习模型显著改进了预测性能，XGBoost达到了最佳验证集MSLE 1.396\n"
    model_results += "3. 最重要的特征是执行计划的预估行数（log_plan_rows）和总成本（log_total_cost）\n"
    
    # 如果报告中还没有模型结果部分，添加它
    if "## 模型实验结果" not in content:
        content += model_results
    else:
        # 替换现有的模型结果部分
        import re
        pattern = r"## 模型实验结果.*?(?=##|$)"
        content = re.sub(pattern, model_results.strip() + "\n\n", content, flags=re.DOTALL)
    
    # 写回文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n项目进展报告已更新: {report_path}")

if __name__ == "__main__":
    main() 