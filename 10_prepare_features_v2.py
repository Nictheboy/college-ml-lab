import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train_transformed.jsonl"
TEST_FILE = DATA_DIR / "test_transformed.jsonl"

class AdvancedFeatureEncoder:
    """改进的特征编码器，包含表-列级别的特征"""
    
    def __init__(self, feature_mapping_path: Path):
        # 加载特征映射
        with open(feature_mapping_path, 'r') as f:
            self.feature_mapping = json.load(f)
        
        # 创建索引映射
        self.table_column_to_idx = {
            pair: idx for idx, pair in enumerate(
                [tuple(p) for p in self.feature_mapping['table_column_pairs']]
            )
        }
        
        self.filter_pattern_to_idx = {
            tuple(pattern): idx for idx, pattern in enumerate(
                self.feature_mapping['filter_patterns']
            )
        }
        
        self.join_pattern_to_idx = {
            (tuple(p[0]), tuple(p[1])): idx 
            for idx, p in enumerate(self.feature_mapping['join_patterns'])
        }
        
        # 基础映射
        self.table_to_id = {
            'title': 0,
            'movie_companies': 1,
            'movie_info_idx': 2,
            'movie_keyword': 3,
            'movie_info': 4,
            'cast_info': 5
        }
        
        self.operator_to_id = {
            '=': 0,
            '>': 1,
            '<': 2,
            '>=': 3,
            '<=': 4,
            '!=': 5
        }
    
    def encode_features(self, item: Dict) -> np.ndarray:
        """将单个查询编码为特征向量"""
        features = []
        
        # 1. 基础特征 (5个)
        features.extend([
            item['features']['num_tables'],
            item['features']['num_joins'],
            item['features']['num_filters'],
            item['features'].get('equality_filters', 0),
            item['features'].get('range_filters', 0)
        ])
        
        # 2. 表特征 (6个)
        table_features = [0] * len(self.table_to_id)
        for table in item['parsed_query']['tables'].values():
            if table in self.table_to_id:
                table_features[self.table_to_id[table]] = 1
        features.extend(table_features)
        
        # 3. 表-列使用特征 (15个)
        table_column_features = [0] * len(self.table_column_to_idx)
        
        # 从过滤条件中提取
        for table, column, _, _ in item['parsed_query']['filter_conditions']:
            key = (table, column)
            if key in self.table_column_to_idx:
                table_column_features[self.table_column_to_idx[key]] = 1
        
        # 从JOIN条件中提取
        for table1, col1, table2, col2 in item['parsed_query']['join_conditions']:
            key1 = (table1, col1)
            key2 = (table2, col2)
            if key1 in self.table_column_to_idx:
                table_column_features[self.table_column_to_idx[key1]] = 1
            if key2 in self.table_column_to_idx:
                table_column_features[self.table_column_to_idx[key2]] = 1
        
        features.extend(table_column_features)
        
        # 4. 过滤模式特征 (27个)
        filter_pattern_features = [0] * len(self.filter_pattern_to_idx)
        for table, column, operator, _ in item['parsed_query']['filter_conditions']:
            key = (table, column, operator)
            if key in self.filter_pattern_to_idx:
                filter_pattern_features[self.filter_pattern_to_idx[key]] += 1
        features.extend(filter_pattern_features)
        
        # 5. JOIN模式特征 (5个)
        join_pattern_features = [0] * len(self.join_pattern_to_idx)
        for table1, col1, table2, col2 in item['parsed_query']['join_conditions']:
            # 标准化JOIN模式
            if table1 < table2:
                key = ((table1, col1), (table2, col2))
            else:
                key = ((table2, col2), (table1, col1))
            
            if key in self.join_pattern_to_idx:
                join_pattern_features[self.join_pattern_to_idx[key]] += 1
        features.extend(join_pattern_features)
        
        # 6. 执行计划特征 (4个)
        plan_features = item['features'].get('plan_features', {})
        features.extend([
            np.log1p(plan_features.get('total_cost', 0)),
            np.log1p(plan_features.get('startup_cost', 0)),
            np.log1p(plan_features.get('plan_rows', 0)),
            plan_features.get('plan_width', 0)
        ])
        
        # 7. 每个表的过滤条件数量 (6个)
        table_filter_counts = [0] * len(self.table_to_id)
        for table, _, _, _ in item['parsed_query']['filter_conditions']:
            if table in self.table_to_id:
                table_filter_counts[self.table_to_id[table]] += 1
        features.extend(table_filter_counts)
        
        # 8. 每个表参与的JOIN数量 (6个)
        table_join_counts = [0] * len(self.table_to_id)
        for table1, _, table2, _ in item['parsed_query']['join_conditions']:
            if table1 in self.table_to_id:
                table_join_counts[self.table_to_id[table1]] += 1
            if table2 in self.table_to_id:
                table_join_counts[self.table_to_id[table2]] += 1
        features.extend(table_join_counts)
        
        # 9. 值范围特征（从过滤条件中提取数值范围）
        numeric_values = []
        for _, column, _, value in item['parsed_query']['filter_conditions']:
            if isinstance(value, (int, float)):
                numeric_values.append(value)
        
        if numeric_values:
            features.extend([
                np.log1p(min(numeric_values)),
                np.log1p(max(numeric_values)),
                np.log1p(np.mean(numeric_values)),
                len(numeric_values)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        
        # 基础特征
        names.extend([
            'num_tables', 'num_joins', 'num_filters', 
            'equality_filters', 'range_filters'
        ])
        
        # 表特征
        for table in self.table_to_id.keys():
            names.append(f'has_{table}')
        
        # 表-列特征
        for table, column in self.feature_mapping['table_column_pairs']:
            names.append(f'uses_{table}_{column}')
        
        # 过滤模式特征
        for table, column, op in self.feature_mapping['filter_patterns']:
            op_name = {
                '=': 'eq', '>': 'gt', '<': 'lt',
                '>=': 'gte', '<=': 'lte', '!=': 'ne'
            }.get(op, op)
            names.append(f'filter_{table}_{column}_{op_name}')
        
        # JOIN模式特征
        for (t1, c1), (t2, c2) in self.feature_mapping['join_patterns']:
            names.append(f'join_{t1}_{c1}__{t2}_{c2}')
        
        # 执行计划特征
        names.extend([
            'log_total_cost', 'log_startup_cost', 
            'log_plan_rows', 'plan_width'
        ])
        
        # 每个表的过滤条件数
        for table in self.table_to_id.keys():
            names.append(f'{table}_filter_count')
        
        # 每个表的JOIN数
        for table in self.table_to_id.keys():
            names.append(f'{table}_join_count')
        
        # 值范围特征
        names.extend([
            'min_filter_value', 'max_filter_value',
            'mean_filter_value', 'num_numeric_filters'
        ])
        
        return names

def load_and_encode_data(file_path: Path, encoder: AdvancedFeatureEncoder) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载并编码数据"""
    X = []
    y = []
    query_ids = []
    
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # 编码特征
            features = encoder.encode_features(item)
            X.append(features)
            
            # 目标值（如果存在）
            if 'target_actual_rows' in item:
                y.append(item['target_actual_rows'])
            
            query_ids.append(item['query_id'])
    
    X = np.array(X)
    y = np.array(y) if y else None
    
    return X, y, query_ids

def main():
    """主函数"""
    print("准备改进的特征数据...")
    
    # 创建特征编码器
    feature_mapping_path = DATA_DIR / 'feature_mapping.json'
    encoder = AdvancedFeatureEncoder(feature_mapping_path)
    
    # 加载训练数据
    print("加载训练数据...")
    X_train, y_train, train_ids = load_and_encode_data(TRAIN_FILE, encoder)
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    X_test, _, test_ids = load_and_encode_data(TEST_FILE, encoder)
    print(f"测试集形状: X={X_test.shape}")
    
    # 获取特征名称
    feature_names = encoder.get_feature_names()
    print(f"\n特征数量: {len(feature_names)}")
    print(f"特征类别分布:")
    print(f"  基础特征: 5")
    print(f"  表特征: 6")
    print(f"  表-列特征: 15")
    print(f"  过滤模式特征: 27")
    print(f"  JOIN模式特征: 5")
    print(f"  执行计划特征: 4")
    print(f"  表级统计特征: 12")
    print(f"  值范围特征: 4")
    print(f"  总计: {5+6+15+27+5+4+12+4}")
    
    # 保存处理后的数据
    print("\n保存特征数据...")
    np.save(DATA_DIR / 'X_train_v2.npy', X_train)
    np.save(DATA_DIR / 'y_train_v2.npy', y_train)
    np.save(DATA_DIR / 'X_test_v2.npy', X_test)
    
    # 保存ID和特征名
    with open(DATA_DIR / 'metadata_v2.pkl', 'wb') as f:
        pickle.dump({
            'train_ids': train_ids,
            'test_ids': test_ids,
            'feature_names': feature_names,
            'feature_mapping': encoder.feature_mapping
        }, f)
    
    print("改进的特征准备完成！")
    
    # 显示一些统计信息
    print(f"\n目标值统计:")
    print(f"  最小值: {y_train.min()}")
    print(f"  最大值: {y_train.max()}")
    print(f"  平均值: {y_train.mean():.2f}")
    print(f"  中位数: {np.median(y_train):.2f}")
    print(f"  标准差: {y_train.std():.2f}")
    
    # 分析特征稀疏性
    print(f"\n特征稀疏性分析:")
    sparsity = (X_train == 0).mean(axis=0)
    print(f"  平均稀疏度: {sparsity.mean():.2%}")
    print(f"  最稀疏的特征: {feature_names[sparsity.argmax()]} ({sparsity.max():.2%})")
    print(f"  最密集的特征: {feature_names[sparsity.argmin()]} ({sparsity.min():.2%})")

if __name__ == "__main__":
    main() 