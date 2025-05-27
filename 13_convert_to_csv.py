import json
import csv
from pathlib import Path

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "predictions.json"
OUTPUT_FILE = DATA_DIR / "predictions.csv"

def convert_json_to_csv():
    """将predictions.json转换为CSV格式"""
    print("转换predictions.json到CSV格式...")
    
    # 读取JSON文件
    with open(INPUT_FILE, 'r') as f:
        predictions = json.load(f)
    
    # 写入CSV文件
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['Query ID', 'Predicted Cardinality'])
        
        # 写入数据
        for pred in predictions:
            writer.writerow([pred['query_id'], pred['predicted_rows']])
    
    print(f"成功转换 {len(predictions)} 条预测结果")
    print(f"输出文件: {OUTPUT_FILE}")
    
    # 显示前几行作为示例
    print("\n前10行预览:")
    with open(OUTPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i < 11:  # 包括表头
                print(line.strip())
            else:
                break
    
    # 验证文件
    with open(OUTPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    print(f"\n文件验证:")
    print(f"总行数: {len(rows)} (包括表头)")
    print(f"数据行数: {len(rows) - 1}")
    
    # 检查是否所有query_id都存在
    query_ids = [int(row[0]) for row in rows[1:]]  # 跳过表头
    expected_ids = list(range(1070))
    
    if set(query_ids) == set(expected_ids):
        print("✓ 所有Query ID (0-1069) 都存在")
    else:
        missing = set(expected_ids) - set(query_ids)
        if missing:
            print(f"✗ 缺少的Query ID: {sorted(missing)}")
        extra = set(query_ids) - set(expected_ids)
        if extra:
            print(f"✗ 多余的Query ID: {sorted(extra)}")

def main():
    """主函数"""
    print("=== JSON到CSV转换工具 ===\n")
    
    # 检查输入文件是否存在
    if not INPUT_FILE.exists():
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return
    
    # 执行转换
    convert_json_to_csv()
    
    print("\n转换完成！")

if __name__ == "__main__":
    main() 