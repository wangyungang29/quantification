# 量化交易系统

一个基于Python的量化交易系统，用于判断股票的涨跌概率和涨跌空间。

## 项目结构

```
quantification/
├── config/              # 配置文件目录
│   └── config.py        # 系统配置
├── data/                # 数据存储目录
├── src/                 # 源代码目录
│   ├── data/            # 数据获取模块
│   │   └── data_fetcher.py  # 从API获取股票数据
│   ├── features/        # 特征提取模块
│   │   └── feature_extractor.py  # 计算技术指标
│   ├── models/          # 模型训练模块
│   │   ├── saved/       # 模型保存目录
│   │   └── model_trainer.py  # 训练预测模型
│   ├── prediction/      # 预测模块
│   │   └── predictor.py # 计算涨跌概率和涨跌空间
│   ├── backtest/        # 回测模块
│   │   └── backtester.py # 评估模型性能
│   └── utils/           # 工具函数目录
├── main.py              # 主脚本
├── requirements.txt     # 依赖文件
├── .env                 # 环境变量文件
└── README.md            # 项目说明
```

## 核心功能

1. **数据获取**：从tushare API获取股票历史数据
2. **特征提取**：计算各种技术指标，如移动平均线、MACD、RSI、KDJ等
3. **模型训练**：使用XGBoost和LightGBM算法训练预测模型
4. **预测功能**：计算股票的涨跌概率和涨跌空间
5. **回测功能**：使用backtrader库模拟交易，评估模型性能

## 安装步骤

1. **克隆项目**：
   ```bash
   git clone <项目地址>
   cd quantification
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **配置API密钥**：
   编辑 `.env` 文件，添加您的tushare API密钥：
   ```
   TUSHARE_TOKEN=your_tushare_token_here
   ```
   您可以访问 [tushare.pro](https://tushare.pro/) 注册账号并获取API密钥。

## 使用方法

### 1. 获取股票数据

```bash
python main.py fetch <股票代码>
```

例如：
```bash
python main.py fetch 600519.SH
```

### 2. 训练模型

```bash
python main.py train <股票代码> --model <模型类型>
```

模型类型可选：`xgboost` 或 `lightgbm`

例如：
```bash
python main.py train 600519.SH --model xgboost
```

### 3. 预测股票

```bash
python main.py predict <股票代码> --model <模型类型>
```

例如：
```bash
python main.py predict 600519.SH --model xgboost
```

### 4. 回测模型

```bash
python main.py backtest <股票代码> <开始日期> <结束日期> --model <模型类型>
```

日期格式：YYYYMMDD

例如：
```bash
python main.py backtest 600519.SH 20230101 20231231 --model xgboost
```

## 技术特点

1. **模块化设计**：各个功能模块独立实现，便于维护和扩展
2. **多种技术指标**：使用TA-Lib库计算多种技术指标，包括：
   - 移动平均线（MA）
   - MACD
   - RSI
   - KDJ
   - 布林带（Bollinger Bands）
   - CCI
   - 威廉指标（WR）
   - 乖离率（BIAS）
3. **机器学习模型**：使用XGBoost和LightGBM等先进的机器学习算法
4. **完整的回测系统**：使用backtrader库进行模拟交易，评估模型性能
5. **命令行界面**：提供友好的命令行接口，方便使用

## 预测结果说明

预测结果包含以下信息：

- **预测方向**：上涨、下跌或横盘
- **置信度**：预测的置信程度
- **涨跌概率**：上涨、横盘、下跌的概率
- **涨跌空间**：基于波动率估计的涨跌空间
- **预期价格**：基于概率和涨跌空间计算的预期价格
- **年化波动率**：风险指标

## 注意事项

1. 您需要注册tushare账号并获取API密钥才能使用数据获取功能
2. 首次使用时，建议先训练模型，然后再进行预测和回测
3. 模型的预测结果仅供参考，实际投资决策需要结合其他因素
4. 本系统仅用于学习和研究目的，不构成投资建议

## 依赖说明

- **核心依赖**：pandas, numpy, scikit-learn, matplotlib, requests
- **数据获取**：tushare
- **技术指标计算**：TA-Lib
- **模型训练**：xgboost, lightgbm
- **回测**：backtrader
- **工具**：joblib, python-dotenv

## 扩展建议

1. **增加更多数据源**：可以添加其他数据源，如聚宽、万得等
2. **优化模型**：可以尝试使用更复杂的模型，如LSTM、Transformer等
3. **增加更多技术指标**：可以添加自定义的技术指标
4. **实盘交易**：可以添加实盘交易功能，对接券商API
5. **可视化界面**：可以添加Web界面，方便用户操作

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎联系项目维护者。