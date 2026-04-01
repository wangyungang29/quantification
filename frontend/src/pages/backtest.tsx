import React, { useState } from 'react';
import { Layout, Menu, Card, Input, Button, Select, Table, Typography, message, DatePicker } from 'antd';
import { Link } from 'react-router-dom';
import axios from 'axios';
import dayjs from 'dayjs';

const { Header, Content } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface BacktestResult {
  initial_capital: number;
  final_value: number;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  average_return: number;
  total_trades: number;
  win_rate: number;
}

const BacktestPage: React.FC = () => {
  const [stockCode, setStockCode] = useState('600519.SH');
  const [modelType, setModelType] = useState('xgboost');
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs('2023-01-01'),
    dayjs('2023-12-31')
  ]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);

  const handleBacktest = async () => {
    setLoading(true);
    try {
      // 调用后端API
      const response = await axios.post('http://localhost:5001/api/backtest', {
        ts_code: stockCode,
        model_type: modelType,
        start_date: dateRange[0].format('YYYYMMDD'),
        end_date: dateRange[1].format('YYYYMMDD')
      });
      setResult(response.data);
      setLoading(false);
    } catch (error) {
      message.error('回测失败，请稍后重试');
      setLoading(false);
    }
  };

  const columns = [
    {
      title: '指标',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
    },
  ];

  const data = result ? [
    { key: '1', name: '初始资金', value: `${result.initial_capital.toFixed(2)}` },
    { key: '2', name: '最终资金', value: `${result.final_value.toFixed(2)}` },
    { key: '3', name: '总收益率', value: `${(result.total_return * 100).toFixed(2)}%` },
    { key: '4', name: '夏普比率', value: `${result.sharpe_ratio.toFixed(2)}` },
    { key: '5', name: '最大回撤', value: `${(result.max_drawdown * 100).toFixed(2)}%` },
    { key: '6', name: '平均收益率', value: `${(result.average_return * 100).toFixed(2)}%` },
    { key: '7', name: '总交易次数', value: `${result.total_trades}` },
    { key: '8', name: '胜率', value: `${(result.win_rate * 100).toFixed(2)}%` },
  ] : [];

  return (
    <Layout>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['3']}>
          <Menu.Item key="1">
            <Link to="/">首页</Link>
          </Menu.Item>
          <Menu.Item key="2">
            <Link to="/predict">预测</Link>
          </Menu.Item>
          <Menu.Item key="3">
            <Link to="/backtest">回测</Link>
          </Menu.Item>
        </Menu>
      </Header>
      <Content style={{ padding: '24px', minHeight: 280 }}>
        <Card>
          <Title level={2}>回测分析</Title>
          <div style={{ marginBottom: 24 }}>
            <Input
              placeholder="输入股票代码，如 600519.SH"
              value={stockCode}
              onChange={(e) => setStockCode(e.target.value)}
              style={{ width: 300, marginRight: 16 }}
            />
            <Select
              defaultValue="xgboost"
              style={{ width: 120, marginRight: 16 }}
              onChange={setModelType}
            >
              <Option value="xgboost">XGBoost</Option>
              <Option value="lightgbm">LightGBM</Option>
            </Select>
            <RangePicker
              value={dateRange}
              onChange={(dates) => setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
              style={{ marginRight: 16 }}
            />
            <Button type="primary" onClick={handleBacktest} loading={loading}>
              回测
            </Button>
          </div>
          {result && (
            <Table columns={columns} dataSource={data} pagination={false} />
          )}
        </Card>
      </Content>
    </Layout>
  );
};

export default BacktestPage;