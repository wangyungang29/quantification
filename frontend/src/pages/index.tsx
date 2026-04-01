import React from 'react';
import { Layout, Menu, Card, Button, Typography } from 'antd';
import { Link } from 'react-router-dom';

const { Header, Content } = Layout;
const { Title } = Typography;

const IndexPage: React.FC = () => {
  return (
    <Layout>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['1']}>
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
          <Title level={2}>量化交易系统</Title>
          <p>这是一个基于Python的量化交易系统，用于判断股票的涨跌概率和涨跌空间。</p>
          <div style={{ marginTop: 24 }}>
            <Button type="primary" size="large">
              <Link to="/predict" style={{ color: 'white' }}>开始预测</Link>
            </Button>
            <Button size="large" style={{ marginLeft: 16 }}>
              <Link to="/backtest">回测分析</Link>
            </Button>
          </div>
        </Card>
      </Content>
    </Layout>
  );
};

export default IndexPage;