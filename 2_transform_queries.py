import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_INPUT_FILE = DATA_DIR / "train_processed.jsonl"
TEST_INPUT_FILE = DATA_DIR / "test_processed.jsonl"
TRAIN_OUTPUT_FILE = DATA_DIR / "train_transformed.jsonl"
TEST_OUTPUT_FILE = DATA_DIR / "test_transformed.jsonl"

@dataclass
class ParsedQuery:
    """存储解析后的查询结构"""
    tables: Dict[str, str]  # alias -> full_name mapping
    join_conditions: List[Tuple[str, str, str, str]]  # (table1, col1, table2, col2)
    filter_conditions: List[Tuple[str, str, str, Any]]  # (table, column, operator, value)
    original_query: str

class SQLParser:
    """SQL查询解析器"""
    
    def __init__(self):
        # 预定义的表名映射
        self.table_mappings = {
            'mc': 'movie_companies',
            't': 'title', 
            'mi_idx': 'movie_info_idx',
            'mk': 'movie_keyword',
            'mi': 'movie_info',
            'ci': 'cast_info'
        }
    
    def parse_sql(self, query: str) -> ParsedQuery:
        """解析SQL查询"""
        query = query.strip().rstrip(';')
        
        # 提取FROM子句中的表和别名
        tables = self._extract_tables(query)
        
        # 提取WHERE子句
        where_clause = self._extract_where_clause(query)
        
        # 解析WHERE条件
        join_conditions, filter_conditions = self._parse_where_conditions(where_clause, tables)
        
        return ParsedQuery(
            tables=tables,
            join_conditions=join_conditions,
            filter_conditions=filter_conditions,
            original_query=query
        )
    
    def _extract_tables(self, query: str) -> Dict[str, str]:
        """提取FROM子句中的表名和别名"""
        # 匹配FROM子句
        from_match = re.search(r'FROM\s+(.*?)\s+WHERE', query, re.IGNORECASE)
        if not from_match:
            return {}
        
        from_clause = from_match.group(1)
        tables = {}
        
        # 解析表名和别名，支持逗号分隔的表
        table_parts = [part.strip() for part in from_clause.split(',')]
        
        for part in table_parts:
            # 匹配 "table_name alias" 格式
            match = re.match(r'(\w+)\s+(\w+)', part.strip())
            if match:
                table_name, alias = match.groups()
                tables[alias] = table_name
            else:
                # 如果没有别名，表名就是别名
                table_name = part.strip()
                tables[table_name] = table_name
        
        return tables
    
    def _extract_where_clause(self, query: str) -> str:
        """提取WHERE子句"""
        where_match = re.search(r'WHERE\s+(.*?)(?:$|;)', query, re.IGNORECASE | re.DOTALL)
        return where_match.group(1).strip() if where_match else ""
    
    def _parse_where_conditions(self, where_clause: str, tables: Dict[str, str]) -> Tuple[List, List]:
        """解析WHERE条件，分离JOIN条件和过滤条件"""
        if not where_clause:
            return [], []
        
        # 按AND分割条件
        conditions = [cond.strip() for cond in re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)]
        
        join_conditions = []
        filter_conditions = []
        
        for condition in conditions:
            if self._is_join_condition(condition, tables):
                join_cond = self._parse_join_condition(condition, tables)
                if join_cond:
                    join_conditions.append(join_cond)
            else:
                filter_cond = self._parse_filter_condition(condition, tables)
                if filter_cond:
                    filter_conditions.append(filter_cond)
        
        return join_conditions, filter_conditions
    
    def _is_join_condition(self, condition: str, tables: Dict[str, str]) -> bool:
        """判断是否为JOIN条件（两个表的列相等）"""
        # 匹配 alias1.col1 = alias2.col2 格式
        pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        match = re.match(pattern, condition.strip())
        
        if match:
            alias1, col1, alias2, col2 = match.groups()
            # 确保两个别名都在表映射中且不同
            return alias1 in tables and alias2 in tables and alias1 != alias2
        
        return False
    
    def _parse_join_condition(self, condition: str, tables: Dict[str, str]) -> Optional[Tuple[str, str, str, str]]:
        """解析JOIN条件"""
        pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        match = re.match(pattern, condition.strip())
        
        if match:
            alias1, col1, alias2, col2 = match.groups()
            table1 = tables.get(alias1, alias1)
            table2 = tables.get(alias2, alias2)
            return (table1, col1, table2, col2)
        
        return None
    
    def _parse_filter_condition(self, condition: str, tables: Dict[str, str]) -> Optional[Tuple[str, str, str, Any]]:
        """解析过滤条件"""
        # 匹配各种操作符
        patterns = [
            (r'(\w+)\.(\w+)\s*=\s*(\d+)', '='),
            (r'(\w+)\.(\w+)\s*>\s*(\d+)', '>'),
            (r'(\w+)\.(\w+)\s*<\s*(\d+)', '<'),
            (r'(\w+)\.(\w+)\s*>=\s*(\d+)', '>='),
            (r'(\w+)\.(\w+)\s*<=\s*(\d+)', '<='),
            (r'(\w+)\.(\w+)\s*!=\s*(\d+)', '!='),
        ]
        
        for pattern, operator in patterns:
            match = re.match(pattern, condition.strip())
            if match:
                alias, column, value = match.groups()
                table = tables.get(alias, alias)
                try:
                    numeric_value = int(value)
                except ValueError:
                    numeric_value = value
                return (table, column, operator, numeric_value)
        
        return None

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        # 预定义的表ID映射
        self.table_to_id = {
            'title': 0,
            'movie_companies': 1,
            'movie_info_idx': 2,
            'movie_keyword': 3,
            'movie_info': 4,
            'cast_info': 5
        }
        
        # 预定义的列ID映射（简化版）
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
        
        # 操作符ID映射
        self.operator_to_id = {
            '=': 0,
            '>': 1,
            '<': 2,
            '>=': 3,
            '<=': 4,
            '!=': 5
        }
    
    def extract_features(self, parsed_query: ParsedQuery, explain_result: Dict) -> Dict[str, Any]:
        """提取查询特征"""
        features = {}
        
        # 基础特征
        features['num_tables'] = len(parsed_query.tables)
        features['num_joins'] = len(parsed_query.join_conditions)
        features['num_filters'] = len(parsed_query.filter_conditions)
        
        # 表特征
        table_ids = [self.table_to_id.get(table, -1) for table in parsed_query.tables.values()]
        features['table_ids'] = sorted(table_ids)
        
        # JOIN特征
        join_features = []
        for table1, col1, table2, col2 in parsed_query.join_conditions:
            t1_id = self.table_to_id.get(table1, -1)
            t2_id = self.table_to_id.get(table2, -1)
            c1_id = self.column_to_id.get(col1, -1)
            c2_id = self.column_to_id.get(col2, -1)
            join_features.append((min(t1_id, t2_id), max(t1_id, t2_id), c1_id, c2_id))
        features['join_features'] = sorted(join_features)
        
        # 过滤条件特征
        filter_features = []
        equality_count = 0
        range_count = 0
        
        for table, column, operator, value in parsed_query.filter_conditions:
            t_id = self.table_to_id.get(table, -1)
            c_id = self.column_to_id.get(column, -1)
            op_id = self.operator_to_id.get(operator, -1)
            
            filter_features.append((t_id, c_id, op_id, value))
            
            if operator == '=':
                equality_count += 1
            elif operator in ['>', '<', '>=', '<=']:
                range_count += 1
        
        features['filter_features'] = filter_features
        features['equality_filters'] = equality_count
        features['range_filters'] = range_count
        
        # 从执行计划提取特征
        plan_features = self._extract_plan_features(explain_result)
        features['plan_features'] = plan_features
        
        return features
    
    def _extract_plan_features(self, explain_result: Dict) -> Dict[str, Any]:
        """从执行计划提取特征"""
        if not explain_result or 'QUERY PLAN' not in explain_result:
            return {}
        
        plan = explain_result['QUERY PLAN'][0]['Plan']
        
        features = {
            'total_cost': plan.get('Total Cost', 0),
            'startup_cost': plan.get('Startup Cost', 0),
            'plan_rows': plan.get('Plan Rows', 0),
            'plan_width': plan.get('Plan Width', 0)
        }
        
        # 递归提取节点类型
        node_types = []
        self._collect_node_types(plan, node_types)
        features['node_types'] = node_types
        
        return features
    
    def _collect_node_types(self, node: Dict, node_types: List[str]):
        """递归收集执行计划中的节点类型"""
        if 'Node Type' in node:
            node_types.append(node['Node Type'])
        
        if 'Plans' in node:
            for child_plan in node['Plans']:
                self._collect_node_types(child_plan, node_types)

def transform_dataset(input_file: Path, output_file: Path, is_train_set: bool):
    """转换数据集"""
    print(f"Transforming {'train' if is_train_set else 'test'} data: {input_file} -> {output_file}")
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    parser = SQLParser()
    extractor = FeatureExtractor()
    
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line.strip())
                
                # 解析SQL查询
                parsed_query = parser.parse_sql(data['query'])
                
                # 提取特征
                features = extractor.extract_features(parsed_query, data.get('explain_result', {}))
                
                # 构建转换后的记录
                transformed_record = {
                    'query_id': data['query_id'],
                    'original_query': data['query'],
                    'parsed_query': {
                        'tables': parsed_query.tables,
                        'join_conditions': parsed_query.join_conditions,
                        'filter_conditions': parsed_query.filter_conditions
                    },
                    'features': features,
                    'explain_result': data.get('explain_result', {})
                }
                
                # 如果是训练数据，保留目标值
                if is_train_set and 'target_actual_rows' in data:
                    transformed_record['target_actual_rows'] = data['target_actual_rows']
                
                outfile.write(json.dumps(transformed_record) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing line {processed_count + 1}: {e}")
                continue
    
    print(f"Successfully transformed {processed_count} items into {output_file}")

def main():
    """主函数"""
    # 确保数据目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 转换训练数据
    transform_dataset(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE, is_train_set=True)
    
    # 转换测试数据
    transform_dataset(TEST_INPUT_FILE, TEST_OUTPUT_FILE, is_train_set=False)
    
    print("Transformation completed!")

if __name__ == "__main__":
    main()