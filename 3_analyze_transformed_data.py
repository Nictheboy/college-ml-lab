import json
from pathlib import Path
from collections import Counter, defaultdict

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train_transformed.jsonl"
TEST_FILE = DATA_DIR / "test_transformed.jsonl"

def load_data(file_path):
    """加载转换后的数据"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_dataset(data, dataset_name):
    """分析数据集统计信息"""
    print(f"\n=== {dataset_name} 数据集分析 ===")
    print(f"总查询数量: {len(data)}")
    
    # 基础统计
    num_tables = [item['features']['num_tables'] for item in data]
    num_joins = [item['features']['num_joins'] for item in data]
    num_filters = [item['features']['num_filters'] for item in data]
    
    print(f"\n表数量统计:")
    print(f"  平均: {sum(num_tables)/len(num_tables):.2f}")
    print(f"  范围: {min(num_tables)} - {max(num_tables)}")
    print(f"  分布: {Counter(num_tables)}")
    
    print(f"\nJOIN数量统计:")
    print(f"  平均: {sum(num_joins)/len(num_joins):.2f}")
    print(f"  范围: {min(num_joins)} - {max(num_joins)}")
    print(f"  分布: {Counter(num_joins)}")
    
    print(f"\n过滤条件数量统计:")
    print(f"  平均: {sum(num_filters)/len(num_filters):.2f}")
    print(f"  范围: {min(num_filters)} - {max(num_filters)}")
    print(f"  分布: {Counter(num_filters)}")
    
    # 表使用统计
    table_usage = Counter()
    for item in data:
        for table in item['parsed_query']['tables'].values():
            table_usage[table] += 1
    
    print(f"\n表使用频率:")
    for table, count in table_usage.most_common():
        print(f"  {table}: {count} ({count/len(data)*100:.1f}%)")
    
    # JOIN模式统计
    join_patterns = Counter()
    for item in data:
        for join in item['parsed_query']['join_conditions']:
            table1, col1, table2, col2 = join
            pattern = f"{table1}.{col1} = {table2}.{col2}"
            join_patterns[pattern] += 1
    
    print(f"\n常见JOIN模式 (前10个):")
    for pattern, count in join_patterns.most_common(10):
        print(f"  {pattern}: {count}")
    
    # 过滤条件统计
    filter_patterns = Counter()
    for item in data:
        for filter_cond in item['parsed_query']['filter_conditions']:
            table, column, operator, value = filter_cond
            pattern = f"{table}.{column} {operator}"
            filter_patterns[pattern] += 1
    
    print(f"\n常见过滤模式 (前10个):")
    for pattern, count in filter_patterns.most_common(10):
        print(f"  {pattern}: {count}")

def compare_datasets(train_data, test_data):
    """比较训练集和测试集的差异"""
    print(f"\n=== 数据集对比分析 ===")
    
    # 表组合对比
    train_table_combos = Counter()
    test_table_combos = Counter()
    
    for item in train_data:
        tables = tuple(sorted(item['parsed_query']['tables'].values()))
        train_table_combos[tables] += 1
    
    for item in test_data:
        tables = tuple(sorted(item['parsed_query']['tables'].values()))
        test_table_combos[tables] += 1
    
    print(f"\n训练集表组合 (前5个):")
    for combo, count in train_table_combos.most_common(5):
        print(f"  {combo}: {count}")
    
    print(f"\n测试集表组合 (前5个):")
    for combo, count in test_table_combos.most_common(5):
        print(f"  {combo}: {count}")
    
    # 查找测试集中训练集没有的表组合
    train_combos_set = set(train_table_combos.keys())
    test_combos_set = set(test_table_combos.keys())
    
    unique_to_test = test_combos_set - train_combos_set
    unique_to_train = train_combos_set - test_combos_set
    
    print(f"\n测试集独有的表组合: {len(unique_to_test)}")
    for combo in list(unique_to_test)[:3]:
        print(f"  {combo}")
    
    print(f"\n训练集独有的表组合: {len(unique_to_train)}")
    for combo in list(unique_to_train)[:3]:
        print(f"  {combo}")

def show_examples(data, dataset_name, num_examples=3):
    """显示转换示例"""
    print(f"\n=== {dataset_name} 转换示例 ===")
    
    for i in range(min(num_examples, len(data))):
        item = data[i]
        print(f"\n示例 {i+1}:")
        print(f"原始查询: {item['original_query']}")
        print(f"表映射: {item['parsed_query']['tables']}")
        print(f"JOIN条件: {item['parsed_query']['join_conditions']}")
        print(f"过滤条件: {item['parsed_query']['filter_conditions']}")
        print(f"特征: 表数={item['features']['num_tables']}, JOIN数={item['features']['num_joins']}, 过滤数={item['features']['num_filters']}")
        if 'target_actual_rows' in item:
            print(f"目标行数: {item['target_actual_rows']}")

def validate_transformation(data, dataset_name):
    """验证转换的正确性"""
    print(f"\n=== {dataset_name} 转换验证 ===")
    
    errors = []
    
    for i, item in enumerate(data):
        # 检查必要字段
        required_fields = ['query_id', 'original_query', 'parsed_query', 'features']
        for field in required_fields:
            if field not in item:
                errors.append(f"行 {i}: 缺少字段 {field}")
        
        # 检查解析结果的一致性
        if 'parsed_query' in item and 'features' in item:
            parsed = item['parsed_query']
            features = item['features']
            
            # 检查表数量一致性
            if len(parsed['tables']) != features['num_tables']:
                errors.append(f"行 {i}: 表数量不一致")
            
            # 检查JOIN数量一致性
            if len(parsed['join_conditions']) != features['num_joins']:
                errors.append(f"行 {i}: JOIN数量不一致")
            
            # 检查过滤条件数量一致性
            if len(parsed['filter_conditions']) != features['num_filters']:
                errors.append(f"行 {i}: 过滤条件数量不一致")
    
    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  {error}")
    else:
        print("转换验证通过，未发现错误！")

def main():
    """主函数"""
    print("加载转换后的数据...")
    
    # 加载数据
    train_data = load_data(TRAIN_FILE)
    test_data = load_data(TEST_FILE)
    
    # 分析各数据集
    analyze_dataset(train_data, "训练集")
    analyze_dataset(test_data, "测试集")
    
    # 对比分析
    compare_datasets(train_data, test_data)
    
    # 显示示例
    show_examples(train_data, "训练集", 2)
    show_examples(test_data, "测试集", 2)
    
    # 验证转换
    validate_transformation(train_data, "训练集")
    validate_transformation(test_data, "测试集")
    
    print(f"\n=== 总结 ===")
    print(f"训练集: {len(train_data)} 个查询")
    print(f"测试集: {len(test_data)} 个查询")
    print("数据转换完成，可以用于机器学习模型训练！")

if __name__ == "__main__":
    main() 