import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train_transformed.jsonl"
TEST_FILE = DATA_DIR / "test_transformed.jsonl"

def analyze_table_column_usage(file_path: Path) -> Dict:
    """分析数据集中表和列的使用情况"""
    table_usage = Counter()
    column_usage = defaultdict(Counter)  # table -> column -> count
    table_column_pairs = Counter()  # (table, column) -> count
    filter_patterns = Counter()  # (table, column, operator) -> count
    join_patterns = Counter()  # ((table1, col1), (table2, col2)) -> count
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # 统计表使用
            for table in item['parsed_query']['tables'].values():
                table_usage[table] += 1
            
            # 统计过滤条件中的列使用
            for table, column, operator, value in item['parsed_query']['filter_conditions']:
                column_usage[table][column] += 1
                table_column_pairs[(table, column)] += 1
                filter_patterns[(table, column, operator)] += 1
            
            # 统计JOIN条件中的列使用
            for table1, col1, table2, col2 in item['parsed_query']['join_conditions']:
                column_usage[table1][col1] += 1
                column_usage[table2][col2] += 1
                table_column_pairs[(table1, col1)] += 1
                table_column_pairs[(table2, col2)] += 1
                
                # 标准化JOIN模式（小表名在前）
                if table1 < table2:
                    join_patterns[((table1, col1), (table2, col2))] += 1
                else:
                    join_patterns[((table2, col2), (table1, col1))] += 1
    
    return {
        'table_usage': table_usage,
        'column_usage': dict(column_usage),
        'table_column_pairs': table_column_pairs,
        'filter_patterns': filter_patterns,
        'join_patterns': join_patterns
    }

def compare_datasets(train_stats: Dict, test_stats: Dict):
    """比较训练集和测试集的统计信息"""
    print("=== 表使用对比 ===")
    
    train_tables = set(train_stats['table_usage'].keys())
    test_tables = set(test_stats['table_usage'].keys())
    
    print(f"训练集使用的表: {sorted(train_tables)}")
    print(f"测试集使用的表: {sorted(test_tables)}")
    print(f"表集合是否相同: {train_tables == test_tables}")
    
    if train_tables != test_tables:
        print(f"仅在训练集中的表: {train_tables - test_tables}")
        print(f"仅在测试集中的表: {test_tables - train_tables}")
    
    print("\n=== 表-列组合对比 ===")
    
    train_pairs = set(train_stats['table_column_pairs'].keys())
    test_pairs = set(test_stats['table_column_pairs'].keys())
    
    print(f"训练集表-列组合数: {len(train_pairs)}")
    print(f"测试集表-列组合数: {len(test_pairs)}")
    print(f"表-列组合是否相同: {train_pairs == test_pairs}")
    
    if train_pairs != test_pairs:
        only_train = train_pairs - test_pairs
        only_test = test_pairs - train_pairs
        print(f"\n仅在训练集中的表-列组合 ({len(only_train)}个):")
        for pair in sorted(list(only_train))[:10]:
            print(f"  {pair}")
        
        print(f"\n仅在测试集中的表-列组合 ({len(only_test)}个):")
        for pair in sorted(list(only_test))[:10]:
            print(f"  {pair}")
    
    # 分析每个表的列使用
    print("\n=== 各表的列使用情况 ===")
    all_tables = sorted(train_tables | test_tables)
    
    for table in all_tables:
        print(f"\n{table}:")
        train_cols = set(train_stats['column_usage'].get(table, {}).keys())
        test_cols = set(test_stats['column_usage'].get(table, {}).keys())
        
        print(f"  训练集列: {sorted(train_cols)}")
        print(f"  测试集列: {sorted(test_cols)}")
        
        if train_cols != test_cols:
            print(f"  仅训练集: {sorted(train_cols - test_cols)}")
            print(f"  仅测试集: {sorted(test_cols - train_cols)}")
    
    # 分析过滤模式
    print("\n=== 过滤模式对比 ===")
    train_filters = set(train_stats['filter_patterns'].keys())
    test_filters = set(test_stats['filter_patterns'].keys())
    
    print(f"训练集过滤模式数: {len(train_filters)}")
    print(f"测试集过滤模式数: {len(test_filters)}")
    
    # 分析JOIN模式
    print("\n=== JOIN模式对比 ===")
    train_joins = set(train_stats['join_patterns'].keys())
    test_joins = set(test_stats['join_patterns'].keys())
    
    print(f"训练集JOIN模式数: {len(train_joins)}")
    print(f"测试集JOIN模式数: {len(test_joins)}")
    print(f"JOIN模式是否相同: {train_joins == test_joins}")
    
    if train_joins != test_joins:
        only_train_joins = train_joins - test_joins
        only_test_joins = test_joins - train_joins
        
        if only_train_joins:
            print(f"\n仅在训练集中的JOIN模式:")
            for join in list(only_train_joins)[:5]:
                print(f"  {join[0]} = {join[1]}")
        
        if only_test_joins:
            print(f"\n仅在测试集中的JOIN模式:")
            for join in list(only_test_joins)[:5]:
                print(f"  {join[0]} = {join[1]}")

def generate_feature_mapping(train_stats: Dict, test_stats: Dict) -> Dict:
    """生成特征映射，包含所有表-列组合"""
    # 收集所有表-列组合
    all_pairs = set(train_stats['table_column_pairs'].keys()) | set(test_stats['table_column_pairs'].keys())
    
    # 按表分组
    table_columns = defaultdict(set)
    for table, column in all_pairs:
        table_columns[table].add(column)
    
    # 生成映射
    feature_mapping = {
        'table_column_pairs': sorted(all_pairs),
        'table_columns': {table: sorted(cols) for table, cols in table_columns.items()},
        'filter_patterns': sorted(set(train_stats['filter_patterns'].keys()) | set(test_stats['filter_patterns'].keys())),
        'join_patterns': sorted(set(train_stats['join_patterns'].keys()) | set(test_stats['join_patterns'].keys()))
    }
    
    return feature_mapping

def main():
    """主函数"""
    print("分析训练集和测试集的表-列使用情况...\n")
    
    # 分析训练集
    print("分析训练集...")
    train_stats = analyze_table_column_usage(TRAIN_FILE)
    
    # 分析测试集
    print("分析测试集...")
    test_stats = analyze_table_column_usage(TEST_FILE)
    
    # 比较数据集
    compare_datasets(train_stats, test_stats)
    
    # 生成特征映射
    print("\n=== 生成特征映射 ===")
    feature_mapping = generate_feature_mapping(train_stats, test_stats)
    
    print(f"总表-列组合数: {len(feature_mapping['table_column_pairs'])}")
    print(f"总过滤模式数: {len(feature_mapping['filter_patterns'])}")
    print(f"总JOIN模式数: {len(feature_mapping['join_patterns'])}")
    
    # 保存特征映射
    output_file = DATA_DIR / 'feature_mapping.json'
    with open(output_file, 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    
    print(f"\n特征映射已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n=== 详细统计 ===")
    for table, columns in sorted(feature_mapping['table_columns'].items()):
        print(f"{table}: {len(columns)} 个列")
        print(f"  列: {columns}")

if __name__ == "__main__":
    main() 