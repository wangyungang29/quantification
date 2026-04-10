import React, { useMemo, useState } from 'react';
import { Layout, Menu, Card, Input, Button, Select, Table, Typography, message } from 'antd';
import { Link } from 'react-router-dom';
import axios from 'axios';

const { Header, Content } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

interface PredictionResult {
  ts_code: string;
  latest_date: string;
  latest_close: number;
  prediction: {
    label: number;
    probability: {
      down: number;
      flat: number;
      up: number;
    };
    price_space: {
      up_space: number;
      down_space: number;
      expected_price: number;
    };
  };
  volatility: number;
}

const PredictPage: React.FC = () => {
  const [stockCode, setStockCode] = useState('600519.SH');
  const [modelType, setModelType] = useState('xgboost');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    try {
      // 调用后端API
      const response = await axios.post('http://localhost:5001/api/predict', { ts_code: stockCode, model_type: modelType });
      setResult(response.data);
      setLoading(false);
    } catch (error) {
      message.error('预测失败，请稍后重试');
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



  const data = useMemo(() => {
console.log('-----')
    if ([undefined, {} as any].includes(result)) {
      return []
    }
    if (result) {
      // 浮动价格
      let floatPrice = result.latest_close;
      // 如果下跌概率大于上涨
      if (result.prediction.probability.down > result.prediction.probability.up) {
        console.log(result.prediction.price_space.down_space,'下跌空间');
        floatPrice = result.latest_close * (1 - result.prediction.price_space.down_space );
      }
      // 如果上涨概率大于下跌
      if (result.prediction.probability.up > result.prediction.probability.down) {
        console.log(result.prediction.price_space.up_space,'上涨空间');
        floatPrice = result.latest_close * (1 + result.prediction.price_space.up_space );
      }
      console.log(floatPrice,'预期价格',result.prediction.probability.up , result.prediction.probability.down);
      return [
        { key: '1', name: '股票代码', value: result.ts_code },
        { key: '2', name: '最新日期', value: result.latest_date },
        { key: '3', name: '最新收盘价', value: `${result.latest_close.toFixed(2)}` },
        { key: '4', name: '预测方向', value: result.prediction.label === 1 ? '上涨' : result.prediction.label === -1 ? '下跌' : '横盘' },
        { key: '5', name: '上涨概率', value: `${(result.prediction.probability.up * 100).toFixed(2)}%` },
        { key: '6', name: '横盘概率', value: `${(result.prediction.probability.flat * 100).toFixed(2)}%` },
        { key: '7', name: '下跌概率', value: `${(result.prediction.probability.down * 100).toFixed(2)}%` },
        { key: '8', name: '上涨空间', value: `${(result.prediction.price_space.up_space * 100).toFixed(2)}%` },
        { key: '9', name: '下跌空间', value: `${(result.prediction.price_space.down_space * 100).toFixed(2)}%` },
        { key: '10', name: '预期价格', value: `${floatPrice.toFixed(2)}` },
        { key: '11', name: '年化波动率', value: `${(result.volatility * 100).toFixed(2)}%` },
      ]
    }

  }, [result]);



  return (
    <Layout>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['2']}>
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
          <Title level={2}>股票预测</Title>
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
            <Button type="primary" onClick={handlePredict} loading={loading}>
              预测
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

export default PredictPage;