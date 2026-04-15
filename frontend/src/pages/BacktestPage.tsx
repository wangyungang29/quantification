import React, { useState } from 'react'
import { Card, Input, Button, Select, Table, Typography, message, DatePicker } from 'antd'
import axios from 'axios'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { Option } = Select
const { RangePicker } = DatePicker

interface Trade {
  date: string
  action: string
  price: number
  shares: number
  amount: number
  commission: number
  total: number
}

interface BacktestResult {
  initial_capital: number
  final_value: number
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  average_return: number
  total_trades: number
  win_rate: number
  trades: Trade[]
}

const BacktestPage: React.FC = () => {
  const [stockCode, setStockCode] = useState('000564.SZ') // 默认设置为供销大集
  const [modelType, setModelType] = useState('xgboost')
  const [strategyType, setStrategyType] = useState('normal') // normal 或 high_return
  // 设置日期范围为三个月前到今天
  const today = dayjs()
  const threeMonthsAgo = dayjs().subtract(3, 'month')
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    threeMonthsAgo,
    today
  ])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<BacktestResult | null>(null)

  const handleBacktest = async () => {
    setLoading(true)
    try {
      let response
      if (strategyType === 'high_return') {
        // 调用高收益策略回测API
        response = await axios.post('http://localhost:5001/api/backtest_500pct', {
          ts_code: stockCode,
          model_type: modelType
        })
      } else {
        // 调用普通策略回测API
        response = await axios.post('http://localhost:5001/api/backtest', {
          ts_code: stockCode,
          model_type: modelType,
          start_date: dateRange[0].format('YYYYMMDD'),
          end_date: dateRange[1].format('YYYYMMDD')
        })
      }
      setResult(response.data)
      setLoading(false)
    } catch (error) {
      message.error('回测失败，请稍后重试')
      setLoading(false)
    }
  }

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
  ]

  const data = result ? [
    { key: '1', name: '初始资金', value: `${result.initial_capital.toFixed(2)}` },
    { key: '2', name: '最终资金', value: `${result.final_value.toFixed(2)}` },
    { key: '3', name: '总收益率', value: `${(result.total_return * 100).toFixed(2)}%` },
    // { key: '4', name: '夏普比率', value: `${result.sharpe_ratio.toFixed(2)}` },
    { key: '5', name: '最大回撤', value: `${(result.max_drawdown * 100).toFixed(2)}%` },
    { key: '6', name: '平均收益率', value: `${(result.average_return * 100).toFixed(2)}%` },
    { key: '7', name: '总交易次数', value: `${result.total_trades}` },
    { key: '8', name: '胜率', value: `${(result.win_rate * 100).toFixed(2)}%` },
  ] : []

  // 交易记录表格列定义
  const tradeColumns = [
    {
      title: '日期',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: '交易类型',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => (
        <span style={{ color: action === '买入' ? 'green' : 'red' }}>
          {action}
        </span>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `${price.toFixed(2)}`,
    },
    {
      title: '数量',
      dataIndex: 'shares',
      key: 'shares',
    },
    {
      title: '金额',
      dataIndex: 'amount',
      key: 'amount',
      render: (amount: number) => `${amount.toFixed(2)}`,
    },
    {
      title: '手续费',
      dataIndex: 'commission',
      key: 'commission',
      render: (commission: number) => `${commission.toFixed(2)}`,
    },
    {
      title: '总计',
      dataIndex: 'total',
      key: 'total',
      render: (total: number) => `${total.toFixed(2)}`,
    },
  ]

  return (
    <Card>
      <Title level={2}>回测分析</Title>
      <div style={{ marginBottom: 24 }}>
        <Input
          placeholder="输入股票代码，如 000564.SZ"
          value={stockCode}
          onChange={(e) => setStockCode(e.target.value)}
          style={{ width: 300, marginRight: 16 }}
        />
        {/* 模型类型 */}
        <Select
          defaultValue="xgboost"
          style={{ width: 120, marginRight: 16 }}
          onChange={setModelType}
        >
          <Option value="xgboost">XGBoost</Option>
          <Option value="lightgbm">LightGBM</Option>
        </Select>
        {/* 策略类型 */}
        <Select
          defaultValue="normal"
          style={{ width: 120, marginRight: 16 }}
          onChange={setStrategyType}
        >
          <Option value="normal">普通策略</Option>
          <Option value="high_return">高收益策略</Option>
        </Select>
        <RangePicker
          value={dateRange}
          onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
          style={{ marginRight: 16 }}
        />
        <Button type="primary" onClick={handleBacktest} loading={loading}>
          回测
        </Button>
      </div>
      {result && (
        <>
          <Table columns={columns} dataSource={data} pagination={false} style={{ marginBottom: 24 }} />
          {result.trades && result.trades.length > 0 && (
            <>
              <Title level={4}>交易记录</Title>
              <Table 
                columns={tradeColumns} 
                dataSource={result.trades.map((trade, index) => ({ ...trade, key: index }))} 
                pagination={false} 
              />
            </>
          )}
        </>
      )}
    </Card>
  )
}

export default BacktestPage