import React, { useState } from 'react'
import { Card, Input, Button, Select, Table, Typography, message, DatePicker } from 'antd'
import axios from 'axios'
import dayjs from 'dayjs'

const { Title, Text } = Typography
const { Option } = Select
const { RangePicker } = DatePicker

interface Trade {
  buy_date: string
  buy_price: number
  buy_signal_type: string
  sell_date: string | null
  sell_price: number | null
  sell_reason: string | null
  quantity: number
  stop_loss: number
  take_profit: number
  profit: number | null
  profit_ratio: number | null
}

interface BacktestResult {
  ts_code: string
  start_date: string
  end_date: string
  initial_capital: number
  final_value: number
  profit: number
  profit_ratio: number
  sharpe_ratio: number
  max_drawdown: number
  total_return: number
  annual_return: number
  total_trades: number
  win_trades: number
  lose_trades: number
  win_rate: number
  model_type: string
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
        response = await axios.post('http://localhost:5001/api/backtest', {
          ts_code: stockCode,
          model_type: modelType,
          start_date: dateRange[0].format('YYYYMMDD'),
          end_date: dateRange[1].format('YYYYMMDD')
        })
      } else {
        // 调用普通策略回测API
        response = await axios.post('http://localhost:5001/api/backtest', {
          ts_code: stockCode,
          model_type: 'chan',
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
    { key: '1', name: '股票代码', value: `${result.ts_code}` },
    { key: '2', name: '初始资金', value: `${result.initial_capital.toFixed(2)}` },
    { key: '3', name: '最终资金', value: `${result.final_value.toFixed(2)}` },
    { key: '4', name: '总收益', value: `${result.profit >= 0 ? '+' : ''}${result.profit.toFixed(2)}` },
    { key: '5', name: '收益率', value: `${result.profit_ratio >= 0 ? '+' : ''}${result.profit_ratio.toFixed(2)}%` },
    { key: '6', name: '总收益率', value: `${result.total_return >= 0 ? '+' : ''}${result.total_return.toFixed(2)}%` },
    { key: '7', name: '年度收益', value: `${result.annual_return >= 0 ? '+' : ''}${result.annual_return.toFixed(2)}%` },
    { key: '8', name: '夏普比率', value: `${result.sharpe_ratio.toFixed(2)}` },
    { key: '9', name: '最大回撤', value: `${result.max_drawdown.toFixed(2)}%` },
    { key: '10', name: '总交易次数', value: `${result.total_trades}` },
    { key: '11', name: '盈利交易', value: `${result.win_trades}` },
    { key: '12', name: '亏损交易', value: `${result.lose_trades}` },
    { key: '13', name: '胜率', value: `${result.win_rate.toFixed(2)}%` },
    { key: '14', name: '模型类型', value: `${result.model_type}` },
  ] : []

  // 交易记录表格列定义
  const tradeColumns = [
    {
      title: '买入日期',
      dataIndex: 'buy_date',
      key: 'buy_date',
    },
    {
      title: '买入价格',
      dataIndex: 'buy_price',
      key: 'buy_price',
      render: (price: number) => `${price.toFixed(2)}`,
    },
    {
      title: '信号类型',
      dataIndex: 'buy_signal_type',
      key: 'buy_signal_type',
      render: (type: string) => (
        <span style={{ color: type.includes('买') ? 'green' : 'red' }}>
          {type}
        </span>
      ),
    },
    {
      title: '卖出日期',
      dataIndex: 'sell_date',
      key: 'sell_date',
      render: (date: string | null) => date || '-',
    },
    {
      title: '卖出价格',
      dataIndex: 'sell_price',
      key: 'sell_price',
      render: (price: number | null) => price ? `${price.toFixed(2)}` : '-',
    },
    {
      title: '卖出原因',
      dataIndex: 'sell_reason',
      key: 'sell_reason',
      render: (reason: string | null) => {
        if (!reason) return '-'
        const color = reason === '止盈' ? 'green' : reason === '止损' ? 'red' : 'orange'
        return <span style={{ color }}>{reason}</span>
      },
    },
    {
      title: '数量',
      dataIndex: 'quantity',
      key: 'quantity',
    },
    {
      title: '止损位',
      dataIndex: 'stop_loss',
      key: 'stop_loss',
      render: (price: number) => `${price.toFixed(2)}`,
    },
    {
      title: '止盈位',
      dataIndex: 'take_profit',
      key: 'take_profit',
      render: (price: number) => `${price.toFixed(2)}`,
    },
    {
      title: '利润',
      dataIndex: 'profit',
      key: 'profit',
      render: (profit: number | null) => {
        if (profit === null) return '-'
        return <span style={{ color: profit >= 0 ? 'green' : 'red' }}>
          {profit >= 0 ? '+' : ''}{profit.toFixed(2)}
        </span>
      },
    },
    {
      title: '利润率',
      dataIndex: 'profit_ratio',
      key: 'profit_ratio',
      render: (ratio: number | null) => {
        if (ratio === null) return '-'
        return <span style={{ color: ratio >= 0 ? 'green' : 'red' }}>
          {ratio >= 0 ? '+' : ''}{ratio.toFixed(2)}%
        </span>
      },
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
          {/* <Option value="high_return">高收益策略</Option> */}
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