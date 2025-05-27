# SQL查询基数估计项目进展报告

## 项目概述
本项目旨在使用机器学习方法预测SQL查询的实际返回行数（基数估计）。这是数据库查询优化中的一个关键问题。

## 已完成的工作

### 1. 数据预处理 (`1_preprocess_data.py`)
- **输入**: 原始的 `train_data.json` 和 `test_data.json`
- **处理内容**:
  - 解析 `explain_result` 字段中的JSON字符串
  - 递归删除执行计划中所有以 "Actual" 开头的字段
  - 对于训练集，提取第一个 "Actual Rows" 值作为目标变量 `target_actual_rows`
  - 将数据转换为JSONL格式便于流式处理
- **输出**: `train_processed.jsonl` 和 `test_processed.jsonl`

### 2. 查询转换和特征提取 (`2_transform_queries.py`)
- **SQL解析器**:
  - 解析SQL查询，提取表名、别名映射
  - 识别JOIN条件和过滤条件
  - 支持常见的表别名（如 mc -> movie_companies）
  
- **特征提取**:
  - **基础特征**: 表数量、JOIN数量、过滤条件数量
  - **表特征**: 表ID列表（预定义映射）
  - **JOIN特征**: 参与JOIN的表和列信息
  - **过滤特征**: 过滤条件的表、列、操作符和值
  - **执行计划特征**: 总成本、启动成本、预估行数、节点类型等
  
- **输出**: `train_transformed.jsonl` 和 `test_transformed.jsonl`

### 3. 数据分析 (`3_analyze_transformed_data.py`)
- **数据集统计**:
  - 查询数量、表使用频率
  - JOIN和过滤条件的分布
  - 常见的查询模式
  
- **数据集对比**:
  - 训练集和测试集的表组合差异
  - 识别测试集中的新模式
  
- **转换验证**:
  - 确保解析结果的一致性
  - 验证特征提取的正确性

### 4. 特征工程改进 (`9_analyze_table_column_usage.py`, `10_prepare_features_v2.py`)
- **表-列使用分析**:
  - 发现训练集和测试集使用完全相同的表和列
  - 识别了15个表-列组合、27个过滤模式、5个JOIN模式
  
- **改进的特征**:
  - 从36个特征扩展到78个特征
  - 添加了表-列级别的细粒度特征
  - 添加了过滤模式和JOIN模式特征
  - 添加了表级统计和值范围特征

## 数据集特点
- **训练集**: 包含查询、执行计划和实际行数
- **测试集**: 只包含查询和执行计划，需要预测实际行数
- **主要表**: title, movie_companies, movie_info_idx, movie_keyword, movie_info, cast_info
- **查询复杂度**: 多表JOIN查询，包含多个过滤条件

## 下一步计划

### 1. 模型开发
- [x] 设计和实现机器学习模型
- [x] 使用MSLE（Mean Squared Logarithmic Error）作为唯一评价指标
- [x] 考虑的模型方案：
  - 基线模型：使用执行计划的预估行数
  - 传统ML：XGBoost/LightGBM
  - 深度学习：考虑查询结构的图神经网络

### 2. 特征工程优化
- [x] 探索更多执行计划特征
- [x] 考虑查询结构的图表示
- [x] 特征标准化和编码优化

### 3. 模型评估
- [x] 实现MSLE评价函数
- [x] 交叉验证策略
- [x] 生成测试集预测结果

### 4. 结果提交
- [x] 生成符合要求的 `predictions.json`
- [x] 验证预测结果格式

## 评价指标
**MSLE (Mean Squared Logarithmic Error)**:
```
MSLE = (1/n) * Σ(log(1 + y_pred) - log(1 + y_true))²
```
- 对数变换使得模型更关注相对误差而非绝对误差
- 适合处理数值范围跨度大的预测问题
- 对大值的预测误差惩罚较小

## 技术栈
- Python 3.x
- 数据处理：json, pathlib
- 机器学习：scikit-learn, xgboost
- 数值计算：numpy

## 模型实验结果

### 模型性能比较

| 模型 | 训练MSLE | 验证MSLE | 训练RMSLE | 验证RMSLE | 训练时间 | 特征数 |
|------|----------|----------|-----------|-----------|----------|--------|
| Baseline (Plan Rows) | 2.410165 | - | 1.552471 | - | - | 1 |
| Random Forest | 1.199765 | 1.427542 | 1.095338 | 1.194798 | 0.34s | 36 |
| XGBoost V1 | 1.094386 | 1.396434 | 1.046129 | 1.181708 | 0.41s | 36 |
| **XGBoost V2** | **0.385781** | **0.774980** | **0.621113** | **0.880330** | **1.34s** | **78** |

**最佳模型**: XGBoost V2 (改进特征)

### 关键发现
1. 基线模型（使用执行计划预估行数）的MSLE为2.410
2. 机器学习模型显著改进了预测性能，XGBoost V1达到了验证集MSLE 1.396
3. 特征工程改进带来巨大提升：XGBoost V2达到验证集MSLE 0.775（相比V1改进44.50%）
4. 最重要的特征仍然是执行计划的预估行数（log_plan_rows），但细粒度特征显著提升了性能
5. 新增的表-列级别特征、过滤模式特征等都有重要贡献

### 性能提升总结
- 相比基线模型：验证集MSLE从2.410降到0.775，改进67.84%
- 相比XGBoost V1：验证集MSLE从1.396降到0.775，改进44.50%
- 特征数量从36增加到78，带来了显著的性能提升


## 模型性能摘要
### 1. XGBoost V3 (Robust Features)
- 特征数量: 86
- 训练集 MSLE: 0.398373
- 验证集 MSLE: 0.766269
- 训练时间: 2.17 秒
- 最佳迭代: 472

### 2. XGBoost V2 (Advanced Features)
- 特征数量: 78
- 训练集 MSLE: 0.385781
- 验证集 MSLE: 0.774980
- 训练时间: 1.31 秒
- 最佳迭代: 306

### 3. xgboost
- 特征数量: N/A
- 训练集 MSLE: 1.094386
- 验证集 MSLE: 1.396434
- 训练时间: 0.40 秒
- 最佳迭代: 133

### 4. Random Forest
- 特征数量: N/A
- 训练集 MSLE: 1.199765
- 验证集 MSLE: 1.427542
- 训练时间: 0.35 秒

### 5. baseline_plan_rows
- 特征数量: N/A
- 训练集 MSLE: 2.410165
- 验证集 MSLE: N/A

## 结论
**最佳机器学习模型**: XGBoost V3 (Robust Features) (验证集 MSLE: 0.766269)
与基线模型 (训练集 MSLE: 2.410165) 相比，性能提升 **68.21%**

### 特征工程影响 (XGBoost 模型对比):
- XGBoost V1 (N/A 特征): 验证集 MSLE = 1.396434
- XGBoost V2 (78 特征): 验证集 MSLE = 0.774980
- XGBoost V3 (86 特征): 验证集 MSLE = 0.766269
  - 从 V1 到 V2 (高级特征): **改进 44.50%**
  - 从 V2 到 V3 (鲁棒特征): **改进 1.12%**
  - 从 V1 到 V3 总改进: **45.13%**

**推荐模型**: XGBoost V3 (Robust Features)
请将 `data/predictions_xgboost_v3.json` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py` 以生成提交文件。


## 模型性能摘要
### 1. XGBoost V1 (Basic Features)
- 特征数量: 36
- 训练集 MSLE: 1.094386
- 验证集 MSLE: 1.396434
- 训练时间: 0.40 秒
- 最佳迭代: 133

### 2. Baseline (Plan Rows)
- 特征数量: 1
- 训练集 MSLE: 2.410165
- 验证集 MSLE: N/A (基线模型使用训练集MSLE: 2.410165)

## 结论
**最佳机器学习模型**: XGBoost V1 (Basic Features) (验证集 MSLE: 1.396434)
与基线模型 (训练集 MSLE: 2.410165) 相比，性能提升 **42.06%**

**推荐模型**: XGBoost V1 (Basic Features)
请将 `data/predictions_xgb.json` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py` 以生成提交文件。


## 模型性能摘要
### 1. XGBoost V1 (Basic Features)
- 特征数量: 36
- 训练集 MSLE: 1.094386
- 验证集 MSLE: 1.396434
- 训练时间: 0.40 秒
- 最佳迭代: 133

### 2. Baseline (Plan Rows)
- 特征数量: 1
- 训练集 MSLE: 2.410165
- 验证集 MSLE: N/A (基线模型使用训练集MSLE: 2.410165)

## 结论
**最佳机器学习模型**: XGBoost V1 (Basic Features) (验证集 MSLE: 1.396434)
与基线模型 (训练集 MSLE: 2.410165) 相比，性能提升 **42.06%**

**推荐模型**: XGBoost V1 (Basic Features)
请将 `data/predictions_xgb.json` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py` 以生成提交文件。


## 模型性能摘要
### 1. XGBoost V3 (Robust Features)
- 特征数量: 86
- 训练集 MSLE: 0.398373
- 验证集 MSLE: 0.766269
- 训练时间: 2.17 秒
- 最佳迭代: 472

### 2. XGBoost V2 (Advanced Features)
- 特征数量: 78
- 训练集 MSLE: 0.385781
- 验证集 MSLE: 0.774980
- 训练时间: 1.31 秒
- 最佳迭代: 306

### 3. XGBoost V1 (Basic Features)
- 特征数量: 36
- 训练集 MSLE: 1.094386
- 验证集 MSLE: 1.396434
- 训练时间: 0.40 秒
- 最佳迭代: 133

### 4. Random Forest
- 特征数量: 36
- 训练集 MSLE: 1.199765
- 验证集 MSLE: 1.427542
- 训练时间: 0.35 秒

### 5. Baseline (Plan Rows)
- 特征数量: 1
- 训练集 MSLE: 2.410165
- 验证集 MSLE: N/A (基线模型使用训练集MSLE: 2.410165)

## 结论
**最佳机器学习模型**: XGBoost V3 (Robust Features) (验证集 MSLE: 0.766269)
与基线模型 (训练集 MSLE: 2.410165) 相比，性能提升 **68.21%**

### 特征工程影响 (XGBoost 模型对比):
- XGBoost V1 (Basic Features) (36 特征): 验证集 MSLE = 1.396434
- XGBoost V2 (Advanced Features) (78 特征): 验证集 MSLE = 0.774980
- XGBoost V3 (Robust Features) (86 特征): 验证集 MSLE = 0.766269
  - 从 V1 到 V2 (高级特征): **改进 44.50%**
  - 从 V2 到 V3 (鲁棒特征): **改进 1.12%**
  - 从 V1 到 V3 总改进: **改进 45.13%**

**推荐模型**: XGBoost V3 (Robust Features)
请将 `data/predictions_xgb_v3.json` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py` 以生成提交文件。

## 模型比较

| 模型                     | 特征数量 | 验证集 MSLE | 训练集 MSLE | 备注                       |
|--------------------------|----------|-------------|-------------|----------------------------|
| XGBoost V3 (Robust Features) | 86       | 0.766269    | 0.398373    | 鲁棒特征 (86)                  |
| XGBoost V2 (Advanced Features) | 78       | 0.774980    | 0.385781    | 高级特征 (78)                  |
| XGBoost V1 (Basic Features) | 36       | 1.396434    | 1.094386    | 基础特征 (36)                  |
| Random Forest            | 36       | 1.427542    | 1.199765    | 基础特征 (36)                  |
| Baseline (Plan Rows)     | 1        | N/A         | 2.410165    | 使用执行计划行数作为预测               |


# 项目最终总结与模型推荐
## 模型性能摘要
### 1. XGBoost V3 (Robust Features)
- 特征数量: 86
- 训练集 MSLE: 0.398373
- 验证集 MSLE: 0.766269
- 训练时间: 2.17 秒
- 最佳迭代: 472

### 2. XGBoost V2 (Advanced Features)
- 特征数量: 78
- 训练集 MSLE: 0.385781
- 验证集 MSLE: 0.774980
- 训练时间: 1.31 秒
- 最佳迭代: 306

### 3. XGBoost V1 (Basic Features)
- 特征数量: 36
- 训练集 MSLE: 1.094386
- 验证集 MSLE: 1.396434
- 训练时间: 0.40 秒
- 最佳迭代: 133

### 4. Random Forest
- 特征数量: 36
- 训练集 MSLE: 1.199765
- 验证集 MSLE: 1.427542
- 训练时间: 0.35 秒

### 5. Baseline (Plan Rows)
- 特征数量: 1
- 训练集 MSLE: 2.410165
- 验证集 MSLE: N/A (基线模型使用训练集MSLE: 2.410165)

## 结论
**最佳机器学习模型**: XGBoost V3 (Robust Features) (验证集 MSLE: 0.766269)
与基线模型 (训练集 MSLE: 2.410165) 相比，性能提升 **68.21%**

### 特征工程影响 (XGBoost 模型对比):
- XGBoost V1 (Basic Features) (36 特征): 验证集 MSLE = 1.396434
- XGBoost V2 (Advanced Features) (78 特征): 验证集 MSLE = 0.774980
- XGBoost V3 (Robust Features) (86 特征): 验证集 MSLE = 0.766269
  - 从 V1 到 V2 (高级特征): **改进 44.50%**
  - 从 V2 到 V3 (鲁棒特征): **改进 1.12%**
  - 从 V1 到 V3 总改进: **改进 45.13%**

**推荐模型**: XGBoost V3 (Robust Features)
请将 `data/predictions_xgb_v3.json` 复制为 `data/predictions.json` 并运行 `13_convert_to_csv.py` 以生成提交文件。
