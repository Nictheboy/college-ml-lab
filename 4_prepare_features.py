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

class FeatureEncoder:
    """将解析后的查询特征编码为数值特征"""
    
    def __init__(self):
        self.table_to_id = {
            'title': 0,
            'movie_companies': 1,
            'movie_info_idx': 2,
            'movie_keyword': 3,
            'movie_info': 4,
            'cast_info': 5
        }
        
        self.column_to_id = {
            'id': 0,
            'movie_id': 1,
            'production_year': 2,
            'kind_id': 3,
            'info_type_id': 4,
            'company_type_id': 5,
            'keyword_id': 6,
            'person_id': 7,
            'role_id': 8
        }
        
        self.operator_to_id = {
            '=': 0,
            '>': 1,
            '<': 2,
            '>=': 3,
            '<=': 4,
            '!=': 5
        }
        
        self.node_type_to_id = {
            'Seq Scan': 0,
            'Index Scan': 1,
            'Index Only Scan': 2,
            'Bitmap Index Scan': 3,
            'Bitmap Heap Scan': 4,
            'Hash Join': 5,
            'Merge Join': 6,
            'Nested Loop': 7,
            'Hash': 8,
            'Sort': 9,
            'Aggregate': 10,
            'Limit': 11
        }
    
    def encode_features(self, item: Dict) -> np.ndarray:
        """将单个查询编码为特征向量"""
        features = []
        
        # 1. 基础特征
        features.extend([
            item['features']['num_tables'],
            item['features']['num_joins'],
            item['features']['num_filters'],
            item['features'].get('equality_filters', 0),
            item['features'].get('range_filters', 0)
        ])
        
        # 2. 表特征 (one-hot encoding)
        table_features = [0] * len(self.table_to_id)
        for table in item['parsed_query']['tables'].values():
            if table in self.table_to_id:
                table_features[self.table_to_id[table]] = 1
        features.extend(table_features)
        
        # 3. 执行计划特征
        plan_features = item['features'].get('plan_features', {})
        features.extend([
            np.log1p(plan_features.get('total_cost', 0)),
            np.log1p(plan_features.get('startup_cost', 0)),
            np.log1p(plan_features.get('plan_rows', 0)),
            plan_features.get('plan_width', 0)
        ])
        
        # 4. 节点类型特征 (count of each type)
        node_type_counts = [0] * len(self.node_type_to_id)
        for node_type in plan_features.get('node_types', []):
            if node_type in self.node_type_to_id:
                node_type_counts[self.node_type_to_id[node_type]] += 1
        features.extend(node_type_counts)
        
        # 5. JOIN类型特征
        join_type_features = [0] * 3  # Hash, Merge, Nested Loop
        for node_type in plan_features.get('node_types', []):
            if 'Hash Join' in node_type:
                join_type_features[0] += 1
            elif 'Merge Join' in node_type:
                join_type_features[1] += 1
            elif 'Nested Loop' in node_type:
                join_type_features[2] += 1
        features.extend(join_type_features)
        
        # 6. 过滤条件统计
        operator_counts = [0] * len(self.operator_to_id)
        for _, _, operator, _ in item['parsed_query']['filter_conditions']:
            if operator in self.operator_to_id:
                operator_counts[self.operator_to_id[operator]] += 1
        features.extend(operator_counts)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = [
            'num_tables', 'num_joins', 'num_filters', 
            'equality_filters', 'range_filters'
        ]
        
        # 表特征名
        for table in self.table_to_id.keys():
            names.append(f'has_{table}')
        
        # 执行计划特征名
        names.extend([
            'log_total_cost', 'log_startup_cost', 
            'log_plan_rows', 'plan_width'
        ])
        
        # 节点类型特征名
        for node_type in self.node_type_to_id.keys():
            names.append(f'node_{node_type.replace(" ", "_").lower()}')
        
        # JOIN类型特征名
        names.extend(['hash_joins', 'merge_joins', 'nested_loops'])
        
        # 操作符特征名
        for op in self.operator_to_id.keys():
            op_name = {
                '=': 'eq', '>': 'gt', '<': 'lt',
                '>=': 'gte', '<=': 'lte', '!=': 'ne'
            }.get(op, op)
            names.append(f'op_{op_name}_count')
        
        return names

def load_and_encode_data(file_path: Path, encoder: FeatureEncoder) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
    print("准备特征数据...")
    
    # 创建特征编码器
    encoder = FeatureEncoder()
    
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
    print(f"前10个特征: {feature_names[:10]}")
    
    # 保存处理后的数据
    print("\n保存特征数据...")
    np.save(DATA_DIR / 'X_train.npy', X_train)
    np.save(DATA_DIR / 'y_train.npy', y_train)
    np.save(DATA_DIR / 'X_test.npy', X_test)
    
    # 保存ID和特征名
    with open(DATA_DIR / 'metadata.pkl', 'wb') as f:
        pickle.dump({
            'train_ids': train_ids,
            'test_ids': test_ids,
            'feature_names': feature_names
        }, f)
    
    print("特征准备完成！")
    
    # 显示一些统计信息
    print(f"\n目标值统计:")
    print(f"  最小值: {y_train.min()}")
    print(f"  最大值: {y_train.max()}")
    print(f"  平均值: {y_train.mean():.2f}")
    print(f"  中位数: {np.median(y_train):.2f}")
    print(f"  标准差: {y_train.std():.2f}")

if __name__ == "__main__":
    main() 