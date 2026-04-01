import React from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Layout, Menu, Card, Button, Typography } from 'antd'
import PredictPage from './pages/PredictPage'
import BacktestPage from './pages/BacktestPage'

const { Header, Content } = Layout
const { Title } = Typography

const HomePage: React.FC = () => {
  return (
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
  )
}

const AppContent: React.FC = () => {
  const location = useLocation()
  
  // 根据当前路径确定选中的菜单项
  const getSelectedKey = () => {
    switch (location.pathname) {
      case '/predict':
        return '2'
      case '/backtest':
        return '3'
      default:
        return '1'
    }
  }
  
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <Menu theme="dark" mode="horizontal" selectedKeys={[getSelectedKey()]}>
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
      <Content style={{ padding: '24px' }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
        </Routes>
      </Content>
    </Layout>
  )
}

const App: React.FC = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  )
}

export default App