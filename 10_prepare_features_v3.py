import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pickle

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train_transformed.jsonl"
TEST_FILE = DATA_DIR / "test_transformed.jsonl"

# Max complexity observed in the original training set (from 3_analyze_transformed_data.py)
MAX_JOINS_TRAIN = 2
MAX_TABLES_TRAIN = 3

class AdvancedFeatureEncoderV3:
    """改进的特征编码器 V3, 包含更鲁棒的复杂性和模式特征"""
    
    def __init__(self, feature_mapping_path: Path):
        with open(feature_mapping_path, 'r') as f:
            self.feature_mapping = json.load(f)
        
        self.table_column_to_idx = {
            tuple(p): idx for idx, p in enumerate(self.feature_mapping['table_column_pairs'])
        }
        self.filter_pattern_to_idx = {
            tuple(p): idx for idx, p in enumerate(self.feature_mapping['filter_patterns'])
        }
        self.join_pattern_to_idx = {
            (tuple(p[0]), tuple(p[1])): idx 
            for idx, p in enumerate(self.feature_mapping['join_patterns'])
        }
        self.table_to_id = {
            'title': 0, 'movie_companies': 1, 'movie_info_idx': 2,
            'movie_keyword': 3, 'movie_info': 4, 'cast_info': 5
        }
    
    def encode_features(self, item: Dict) -> np.ndarray:
        features = []
        parsed_query = item['parsed_query']
        base_features = item['features']
        plan_features_dict = base_features.get('plan_features', {})

        # 1. 基础特征 (5)
        num_tables = base_features['num_tables']
        num_joins = base_features['num_joins']
        num_filters = base_features['num_filters']
        equality_filters = base_features.get('equality_filters', 0)
        range_filters = base_features.get('range_filters', 0)
        features.extend([num_tables, num_joins, num_filters, equality_filters, range_filters])

        # 2. 表特征 (6)
        table_presence = [0] * len(self.table_to_id)
        for table in parsed_query['tables'].values():
            if table in self.table_to_id:
                table_presence[self.table_to_id[table]] = 1
        features.extend(table_presence)

        # 3. 表-列使用特征 (15)
        table_column_usage = [0] * len(self.table_column_to_idx)
        active_table_columns = set()
        for table, column, _, _ in parsed_query['filter_conditions']:
            active_table_columns.add((table, column))
        for t1, c1, t2, c2 in parsed_query['join_conditions']:
            active_table_columns.add((t1, c1))
            active_table_columns.add((t2, c2))
        for pair in active_table_columns:
            if pair in self.table_column_to_idx:
                table_column_usage[self.table_column_to_idx[pair]] = 1
        features.extend(table_column_usage)

        # 4. 过滤模式特征 (27)
        filter_pattern_counts = [0] * len(self.filter_pattern_to_idx)
        for table, column, operator, _ in parsed_query['filter_conditions']:
            key = (table, column, operator)
            if key in self.filter_pattern_to_idx:
                filter_pattern_counts[self.filter_pattern_to_idx[key]] += 1
        features.extend(filter_pattern_counts)

        # 5. JOIN模式特征 (5)
        join_pattern_counts = [0] * len(self.join_pattern_to_idx)
        for t1, c1, t2, c2 in parsed_query['join_conditions']:
            key = tuple(sorted(((t1, c1), (t2, c2)))) # Ensure consistent ordering for lookup
            if key in self.join_pattern_to_idx:
                 join_pattern_counts[self.join_pattern_to_idx[key]] +=1
            elif (key[1],key[0]) in self.join_pattern_to_idx: # check reverse due to original mapping construction
                 join_pattern_counts[self.join_pattern_to_idx[(key[1],key[0])]] +=1
        features.extend(join_pattern_counts)

        # 6. 执行计划特征 (4)
        features.extend([
            np.log1p(plan_features_dict.get('total_cost', 0)),
            np.log1p(plan_features_dict.get('startup_cost', 0)),
            np.log1p(plan_features_dict.get('plan_rows', 0)),
            plan_features_dict.get('plan_width', 0)
        ])

        # 7. 每个表的过滤条件数量 (6)
        table_filter_counts = [0] * len(self.table_to_id)
        for table, _, _, _ in parsed_query['filter_conditions']:
            if table in self.table_to_id:
                table_filter_counts[self.table_to_id[table]] += 1
        features.extend(table_filter_counts)

        # 8. 每个表参与的JOIN数量 (6)
        table_join_counts = [0] * len(self.table_to_id)
        for t1, _, t2, _ in parsed_query['join_conditions']:
            if t1 in self.table_to_id: table_join_counts[self.table_to_id[t1]] += 1
            if t2 in self.table_to_id: table_join_counts[self.table_to_id[t2]] += 1
        features.extend(table_join_counts)

        # 9. 值范围特征 (4)
        numeric_values = [val for _, _, _, val in parsed_query['filter_conditions'] if isinstance(val, (int, float))]
        if numeric_values:
            safe_min = np.log1p(np.maximum(0, min(numeric_values)))
            safe_max = np.log1p(np.maximum(0, max(numeric_values)))
            safe_mean = np.log1p(np.maximum(0, np.mean(numeric_values)))
            features.extend([safe_min, safe_max, safe_mean, len(numeric_values)])
        else:
            features.extend([0,0,0,0])

        # --- V3 New Features ---
        # 10. Complexity Flags (2)
        features.append(1 if num_joins > MAX_JOINS_TRAIN else 0)
        features.append(1 if num_tables > MAX_TABLES_TRAIN else 0)

        # 11. Complexity Interaction (1)
        features.append(num_tables * num_joins)

        # 12. Generalized Pattern Counts (3)
        distinct_tables_in_joins = set()
        for t1, _, t2, _ in parsed_query['join_conditions']:
            distinct_tables_in_joins.add(t1)
            distinct_tables_in_joins.add(t2)
        features.append(len(distinct_tables_in_joins))
        features.append(equality_filters / (num_filters + 1e-6))
        features.append(range_filters / (num_filters + 1e-6))

        # 13. Specific Column Filter Flags (2) - simplified
        has_title_pyear_filter = 0
        has_title_kind_id_filter = 0
        for table, column, _, _ in parsed_query['filter_conditions']:
            if table == 'title' and column == 'production_year': has_title_pyear_filter = 1
            if table == 'title' and column == 'kind_id': has_title_kind_id_filter = 1
        features.extend([has_title_pyear_filter, has_title_kind_id_filter])
        
        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        names = []
        names.extend(['num_tables', 'num_joins', 'num_filters', 'equality_filters', 'range_filters'])
        for table in self.table_to_id: names.append(f'has_{table}')
        for table, column in self.feature_mapping['table_column_pairs']: names.append(f'uses_{table}_{column}')
        for table, column, op in self.feature_mapping['filter_patterns']:
            op_name = {'=': 'eq', '>': 'gt', '<': 'lt', '>=': 'gte', '<=': 'lte', '!=': 'ne'}.get(op,op)
            names.append(f'filter_{table}_{column}_{op_name}')
        for (t1,c1),(t2,c2) in self.feature_mapping['join_patterns']: names.append(f'join_{t1}_{c1}__{t2}_{c2}')
        names.extend(['log_total_cost', 'log_startup_cost', 'log_plan_rows', 'plan_width'])
        for table in self.table_to_id: names.append(f'{table}_filter_count')
        for table in self.table_to_id: names.append(f'{table}_join_count')
        names.extend(['min_filter_val_log1p', 'max_filter_val_log1p', 'mean_filter_val_log1p', 'num_numeric_filters'])
        # V3 Names
        names.extend(['exceeds_train_max_joins', 'exceeds_train_max_tables'])
        names.append('tables_X_joins')
        names.extend(['num_distinct_tables_in_joins', 'filter_equality_ratio', 'filter_range_ratio'])
        names.extend(['has_filter_title_pyear', 'has_filter_title_kind_id'])
        return names

def load_and_encode_data(file_path: Path, encoder: AdvancedFeatureEncoderV3) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, query_ids = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            features = encoder.encode_features(item)
            X.append(features)
            if 'target_actual_rows' in item: y.append(item['target_actual_rows'])
            query_ids.append(item['query_id'])
    return np.array(X), np.array(y) if y else None, query_ids

def main():
    print("准备改进的特征数据 V3...")
    feature_mapping_path = DATA_DIR / 'feature_mapping.json'
    if not feature_mapping_path.exists():
        print(f"错误: 特征映射文件 {feature_mapping_path} 未找到. 请先运行 9_analyze_table_column_usage.py")
        return
    encoder = AdvancedFeatureEncoderV3(feature_mapping_path)
    
    print("加载训练数据...")
    X_train, y_train, train_ids = load_and_encode_data(TRAIN_FILE, encoder)
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    
    print("\n加载测试数据...")
    X_test, _, test_ids = load_and_encode_data(TEST_FILE, encoder)
    print(f"测试集形状: X={X_test.shape}")
    
    feature_names = encoder.get_feature_names()
    print(f"\n特征数量: {len(feature_names)} (旧: 78, 新增: {len(feature_names)-78})")
    
    # Verify feature vector length
    if X_train.shape[1] != len(feature_names):
        print(f"错误: 特征向量长度 ({X_train.shape[1]}) 与特征名列表长度 ({len(feature_names)}) 不匹配!")
        # This part helps debug name list generation vs actual encoding
        # print("Expected names based on get_feature_names():")
        # for i, name in enumerate(feature_names):
        #     print(f"  {i}: {name}")
        # print("Actual encoded sample (first 10 values from first training sample):")
        # print(X_train[0][:10])
        # print("Last 10 values:")
        # print(X_train[0][-10:])
        return

    print("\n保存特征数据 V3...")
    np.save(DATA_DIR / 'X_train_v3.npy', X_train)
    np.save(DATA_DIR / 'y_train_v3.npy', y_train)
    np.save(DATA_DIR / 'X_test_v3.npy', X_test)
    
    with open(DATA_DIR / 'metadata_v3.pkl', 'wb') as f:
        pickle.dump({
            'train_ids': train_ids, 'test_ids': test_ids,
            'feature_names': feature_names, 'feature_mapping': encoder.feature_mapping
        }, f)
    print("改进的特征准备 V3 完成！")

if __name__ == "__main__":
    main() 