import json
from pathlib import Path
# No pandas needed for this script based on its current functionality.
# import pandas as pd 

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "report"
PROGRESS_FILE = REPORT_DIR / "project_progress.md"

# Consistent model names and default feature counts if not in JSON
MODEL_INFO = {
    'baseline': {'display_name': 'Baseline (Plan Rows)', 'default_features': 1, 'file_key': 'baseline'},
    'random_forest': {'display_name': 'Random Forest', 'default_features': 36, 'file_key': 'rf'},
    'xgboost_v1': {'display_name': 'XGBoost V1 (Basic Features)', 'default_features': 36, 'file_key': 'xgb'}, # xgb_results.json
    'xgboost_v2': {'display_name': 'XGBoost V2 (Advanced Features)', 'default_features': 78, 'file_key': 'xgb_v2'},
    'xgboost_v3': {'display_name': 'XGBoost V3 (Robust Features)', 'default_features': 86, 'file_key': 'xgb_v3'}
}

def load_results_data():
    """加载所有模型的结果JSON文件"""
    results_files = {}
    for key, info in MODEL_INFO.items():
        file_key = info['file_key']
        # Default naming convention using file_key
        results_files[key] = DATA_DIR / f"{file_key}_results.json"

    # Special case for the very first XGBoost model which was just xgb_results.json
    # If the model key is 'xgboost_v1' and its expected file (xgb_results.json based on file_key='xgb') 
    # doesn't exist but an older general 'xgb_results.json' does, this might be redundant
    # as the file_key 'xgb' should already point to it. This check is more for clarity or if 
    # file_key for xgboost_v1 was initially 'xgboost_v1'. With file_key='xgb', it should find 'xgb_results.json'.
    # Let's ensure it correctly picks up "xgb_results.json" for "xgboost_v1"
    if MODEL_INFO['xgboost_v1']['file_key'] == 'xgb':
         results_files['xgboost_v1'] = DATA_DIR / 'xgb_results.json'
    
    loaded_results = {}
    for name, path in results_files.items(): # name here is the primary key like 'xgboost_v1'
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                data['model_key'] = name # Store the primary key for mapping back to MODEL_INFO
                loaded_results[name] = data
        else:
            print(f"警告: 结果文件 {path} 未找到 for model key {name} (expected from file_key: {MODEL_INFO[name]['file_key']})。")
    return loaded_results

def generate_summary_text(loaded_results: dict) -> str:
    """生成项目总结文本"""
    summary_lines = ["# 项目最终总结与模型推荐\n"]
    summary_lines.append("## 模型性能摘要\n")
    
    all_processed_results = []
    for model_primary_key, res_data in loaded_results.items(): # model_primary_key is 'baseline', 'xgboost_v1' etc.
        processed_res = res_data.copy()
        # processed_res['model_key'] is already set during load_results_data
        if model_primary_key == 'baseline':
            processed_res['val_msle_for_sort'] = res_data.get('train_msle', float('inf'))
        else:
            processed_res['val_msle_for_sort'] = res_data.get('val_msle', float('inf'))
        all_processed_results.append(processed_res)

    all_processed_results.sort(key=lambda x: x['val_msle_for_sort'])

    rank = 1
    for res in all_processed_results:
        model_key_from_res = res['model_key'] # This is the primary key like 'xgboost_v1'
        display_name = MODEL_INFO.get(model_key_from_res, {}).get('display_name', model_key_from_res)
        num_features = res.get('num_features', MODEL_INFO.get(model_key_from_res, {}).get('default_features', 'N/A'))

        summary_lines.append(f"### {rank}. {display_name}\n")
        summary_lines.append(f"- 特征数量: {num_features}\n")
        summary_lines.append(f"- 训练集 MSLE: {res.get('train_msle', 'N/A'):.6f}\n")
        
        if model_key_from_res != 'baseline' and 'val_msle' in res:
            summary_lines.append(f"- 验证集 MSLE: {res['val_msle']:.6f}\n")
        elif model_key_from_res == 'baseline':
             summary_lines.append(f"- 验证集 MSLE: N/A (基线模型使用训练集MSLE: {res.get('train_msle', 'N/A'):.6f})\n")
        else:
            summary_lines.append(f"- 验证集 MSLE: N/A\n")

        if 'training_time' in res: summary_lines.append(f"- 训练时间: {res.get('training_time'):.2f} 秒\n")
        if 'best_iteration' in res: summary_lines.append(f"- 最佳迭代: {res.get('best_iteration')}\n")
        summary_lines.append("\n")
        rank += 1
        
    summary_lines.append("## 结论\n")
    
    ml_models = [r for r in all_processed_results if r['model_key'] != 'baseline' and 'val_msle' in r and r['val_msle'] != float('inf')]
    baseline_res = loaded_results.get('baseline')
    best_ml_model = ml_models[0] if ml_models else None

    if best_ml_model:
        best_ml_model_display_name = MODEL_INFO.get(best_ml_model['model_key'], {}).get('display_name', best_ml_model['model_key'])
        summary_lines.append(f"**最佳机器学习模型**: {best_ml_model_display_name} "
                             f"(验证集 MSLE: {best_ml_model['val_msle']:.6f})\n")
        if baseline_res and 'train_msle' in baseline_res:
            baseline_msle_val = baseline_res['train_msle']
            improvement_over_baseline = (baseline_msle_val - best_ml_model['val_msle']) / baseline_msle_val * 100
            summary_lines.append(f"与基线模型 (训练集 MSLE: {baseline_msle_val:.6f}) 相比，性能提升 **{improvement_over_baseline:.2f}%**\n")
    elif baseline_res: 
        summary_lines.append(f"仅找到基线模型结果。基线模型MSLE (训练集): {baseline_res.get('train_msle', 'N/A'):.6f}\n")
    else: 
        summary_lines.append("没有找到任何模型结果。\n")
        return "".join(summary_lines)

    xgb_v1 = loaded_results.get('xgboost_v1')
    xgb_v2 = loaded_results.get('xgboost_v2')
    xgb_v3 = loaded_results.get('xgboost_v3')

    if xgb_v1 and xgb_v2 and xgb_v3 and all('val_msle' in d for d in [xgb_v1, xgb_v2, xgb_v3]):
        summary_lines.append("\n### 特征工程影响 (XGBoost 模型对比):\n")
        for key_in_loop, data in [('xgboost_v1', xgb_v1), ('xgboost_v2', xgb_v2), ('xgboost_v3', xgb_v3)]:
            # key_in_loop is 'xgboost_v1', 'xgboost_v2', etc.
            name = MODEL_INFO[key_in_loop]['display_name']
            features = data.get('num_features', MODEL_INFO[key_in_loop]['default_features'])
            msle = data['val_msle']
            summary_lines.append(f"- {name} ({features} 特征): 验证集 MSLE = {msle:.6f}\n")
        
        xgb_v1_msle = xgb_v1['val_msle']; xgb_v2_msle = xgb_v2['val_msle']; xgb_v3_msle = xgb_v3['val_msle']
        imp_v1_to_v2 = (xgb_v1_msle - xgb_v2_msle) / xgb_v1_msle * 100
        imp_v2_to_v3 = (xgb_v2_msle - xgb_v3_msle) / xgb_v2_msle * 100
        imp_v1_to_v3 = (xgb_v1_msle - xgb_v3_msle) / xgb_v1_msle * 100
        summary_lines.append(f"  - 从 V1 到 V2 (高级特征): **改进 {imp_v1_to_v2:.2f}%**\n")
        summary_lines.append(f"  - 从 V2 到 V3 (鲁棒特征): **{ '改进 ' + format(imp_v2_to_v3, '.2f') + '%' if imp_v2_to_v3 > 0 else '性能下降 ' + format(-imp_v2_to_v3, '.2f') + '%' }**\n")
        summary_lines.append(f"  - 从 V1 到 V3 总改进: **改进 {imp_v1_to_v3:.2f}%**\n")

    if best_ml_model:
        recommended_model_key = best_ml_model['model_key']
        recommended_model_display_name = MODEL_INFO.get(recommended_model_key, {}).get('display_name', recommended_model_key)
        summary_lines.append(f"\n**推荐模型**: {recommended_model_display_name}\n")
        
        # Use file_key for constructing the prediction filename to copy
        prediction_file_actual_key = MODEL_INFO[recommended_model_key]['file_key']
        prediction_file_to_copy = f"predictions_{prediction_file_actual_key}.json"
        
        summary_lines.append(f"请将 `data/{prediction_file_to_copy}` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py` 以生成提交文件。\n")
    elif baseline_res:
        summary_lines.append(f"\n**推荐模型**: {MODEL_INFO['baseline']['display_name']} (基于可用数据)\n")
        # Baseline prediction file is standard
        summary_lines.append(f"请将 `data/predictions_baseline.json` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py`。\n")
    return "".join(summary_lines)

def update_project_progress(summary_text: str, loaded_results: dict):
    """更新项目进展Markdown文件"""
    table_lines = ["\n## 模型比较\n"]
    table_lines.append("| 模型                     | 特征数量 | 验证集 MSLE | 训练集 MSLE | 备注                       |")
    table_lines.append("|--------------------------|----------|-------------|-------------|----------------------------|")

    all_processed_results_table = []
    for model_primary_key_table, res_data_table in loaded_results.items():
        processed_res_table = res_data_table.copy()
        # processed_res_table['model_key'] is already set
        if model_primary_key_table == 'baseline':
            processed_res_table['val_msle_for_sort'] = res_data_table.get('train_msle', float('inf'))
        else:
            processed_res_table['val_msle_for_sort'] = res_data_table.get('val_msle', float('inf'))
        all_processed_results_table.append(processed_res_table)
    all_processed_results_table.sort(key=lambda x: x['val_msle_for_sort'])

    for res in all_processed_results_table:
        model_key_from_res_table = res['model_key'] # This is primary key like 'xgboost_v1'
        display_name = MODEL_INFO.get(model_key_from_res_table, {}).get('display_name', model_key_from_res_table)
        num_features = res.get('num_features', MODEL_INFO.get(model_key_from_res_table, {}).get('default_features', 'N/A'))
        
        val_msle_val = res.get('val_msle', 'N/A')
        val_msle_str = f"{val_msle_val:.6f}" if isinstance(val_msle_val, float) else ('N/A' if model_key_from_res_table != 'baseline' else 'N/A (见训练集)')
        if model_key_from_res_table == 'baseline': val_msle_str = 'N/A' # Explicitly N/A for baseline val_msle in table

        train_msle_val = res.get('train_msle', 'N/A')
        train_msle_str = f"{train_msle_val:.6f}" if isinstance(train_msle_val, float) else 'N/A'
        
        notes = ""
        # model_key_for_notes is the primary key like 'xgboost_v1'
        model_key_for_notes = res.get('model_key') 
        if model_key_for_notes == 'baseline': notes = "使用执行计划行数作为预测"
        # Use num_features determined above for notes
        elif model_key_for_notes == 'xgboost_v1': notes = f"基础特征 ({num_features})"
        elif model_key_for_notes == 'xgboost_v2': notes = f"高级特征 ({num_features})"
        elif model_key_for_notes == 'xgboost_v3': notes = f"鲁棒特征 ({num_features})"
        elif model_key_for_notes == 'random_forest': notes = f"基础特征 ({num_features})"
        
        table_lines.append(f"| {display_name:<24} | {str(num_features):<8} | {val_msle_str:<11} | {train_msle_str:<11} | {notes:<26} |")

    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f: content = f.read()
    except FileNotFoundError: content = "# 项目进展报告\n"

    table_section_marker = "## 模型比较"
    if table_section_marker in content:
        start_index = content.find(table_section_marker)
        next_section_marker = "\n## "
        end_of_table_index = content.find(next_section_marker, start_index + len(table_section_marker))
        content = content[:start_index] + (content[end_of_table_index:] if end_of_table_index != -1 else "")
    
    content = content.strip() + "\n" + "\n".join(table_lines) + "\n"

    summary_section_marker = "# 项目最终总结与模型推荐"
    if summary_section_marker in content:
        content = content.split(summary_section_marker)[0].strip()
    content = content + "\n\n" + summary_text

    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f: f.write(content)
    print(f"项目进展报告已更新: {PROGRESS_FILE}")

def main():
    """主函数：加载数据，生成总结，更新报告"""
    print("生成最终总结报告...")
    
    loaded_results = load_results_data()
    if not loaded_results:
        print("未能加载任何模型结果，无法生成总结。")
        return

    summary_text_for_print_and_report = generate_summary_text(loaded_results)
    
    print("\n" + "="*30 + " 终端输出总结 " + "="*30 + "\n")
    print(summary_text_for_print_and_report)
    print("\n" + "="*70 + "\n")

    update_project_progress(summary_text_for_print_and_report, loaded_results)
    
    # Recommendation for next step based on the best model found
    # This logic is now part of generate_summary_text, but can be reiterated here for clarity if needed
    # For now, the printout from generate_summary_text is sufficient for guiding the user.

if __name__ == "__main__":
    main() 