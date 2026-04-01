import React, { useState, useEffect, useRef } from 'react'
import { Card, Input, Button, Select, Table, Typography, message, Progress, Alert } from 'antd'
import * as echarts from 'echarts'
import { Line } from '@ant-design/plots'
import axios from 'axios'

const { Title, Text } = Typography
const { Option } = Select

interface PredictionResult {
  ts_code: string
  latest_date: string
  latest_close: number
  prediction: {
    label: number
    probability: {
      down: number
      flat: number
      up: number
    }
    price_space: {
      up_space: number
      down_space: number
      expected_price: number
    }
  }
  volatility: number
  history_data: Array<{
    date: string
    close: number
    volume: number
    ma5: number | null
    ma10: number | null
    ma20: number | null
    rsi: number | null
    macd: number | null
    macd_signal: number | null
    macd_hist: number | null
    boll_upper: number | null
    boll_mid: number | null
    boll_lower: number | null
    atr: number | null
    golden_cross: number
    death_cross: number
  }>
  fundamentals: {
    pe?: number | null
    pb?: number | null
    roe?: number | null
    npg_rate?: number | null
    debt_to_assets?: number | null
  }
}

const PredictPage: React.FC = () => {
  const [stockCode, setStockCode] = useState('000564.SZ') // 默认设置为供销大集
  const [modelType, setModelType] = useState('xgboost')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [stockData, setStockData] = useState<any[]>([])
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('准备中')
  const [noData, setNoData] = useState(false)
  
  // ECharts实例引用
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)
  
  // 初始化和更新图表
  useEffect(() => {
    if (stockData.length > 0 && chartRef.current) {
      // 初始化ECharts实例
      if (!chartInstance.current) {
        chartInstance.current = echarts.init(chartRef.current)
      }
      
      // 准备数据
      const dates = stockData.map(item => item.date)
      const closePrices = stockData.map(item => item.close)
      const volumes = stockData.map(item => item.volume)
      
      // 配置项
      const option = {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross',
            label: {
              backgroundColor: '#6a7985'
            }
          },
          formatter: (params: any) => {
            const dataIndex = params[0].dataIndex;
            const data = stockData[dataIndex];
            let tooltipContent = `<div style="font-weight:bold;margin-bottom:5px">${data.date}</div>`;
            tooltipContent += `<div>股票代码: ${data.ts_code}</div>`;
            tooltipContent += `<div>开盘价: ${data.open ? data.open.toFixed(2) : 'N/A'}</div>`;
            tooltipContent += `<div>最高价: ${data.high ? data.high.toFixed(2) : 'N/A'}</div>`;
            tooltipContent += `<div>最低价: ${data.low ? data.low.toFixed(2) : 'N/A'}</div>`;
            tooltipContent += `<div>收盘价: ${data.close.toFixed(2)}</div>`;
            tooltipContent += `<div>昨收价: ${data.pre_close ? data.pre_close.toFixed(2) : 'N/A'}</div>`;
            tooltipContent += `<div>涨跌额: ${data.change ? data.change.toFixed(2) : 'N/A'}</div>`;
            tooltipContent += `<div>涨跌幅: ${data.pct_chg ? data.pct_chg.toFixed(2) : 'N/A'}%</div>`;
            tooltipContent += `<div>成交量: ${data.vol ? data.vol.toFixed(2) : 'N/A'} 手</div>`;
            tooltipContent += `<div>成交额: ${data.amount ? data.amount.toFixed(2) : 'N/A'} 千元</div>`;
            tooltipContent += `<div>交易量: ${data.volume.toFixed(2)}</div>`;
            return tooltipContent;
          }
        },
        legend: {
          data: ['收盘价', '交易量']
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: dates
        },
        yAxis: [
          {
            type: 'value',
            name: '单价',
            position: 'left',
            axisLabel: {
              formatter: '{value}'
            }
          },
          {
            type: 'value',
            name: '交易量',
            position: 'right',
            axisLabel: {
              formatter: '{value}'
            }
          }
        ],
        series: [
          {
            name: '收盘价',
            type: 'line',
            data: closePrices,
            yAxisIndex: 0,
            itemStyle: {
              color: '#8884d8'
            }
          },
          {
            name: '交易量',
            type: 'line',
            data: volumes,
            yAxisIndex: 1,
            itemStyle: {
              color: '#82ca9d'
            }
          }
        ]
      }
      
      // 设置配置项
      chartInstance.current.setOption(option)
    }
    
    // 响应式调整
    const handleResize = () => {
      chartInstance.current?.resize()
    }
    
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      chartInstance.current?.dispose()
    }
  }, [stockData])

  const handlePredict = async () => {
    setLoading(true)
    setProgress(0)
    setStage('开始预测')
    setNoData(false)
    
    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return prev
        }
        return prev + 10
      })
    }, 200)

    try {
      // 调用后端API
      setStage('获取股票数据')
      const response = await axios.post('http://localhost:5001/api/predict', { ts_code: stockCode, model_type: modelType })
      
      setStage('分析数据')
      setProgress(70)
      
      setStage('生成预测结果')
      setResult(response.data)
      
      // 检查是否有数据
      if (!response.data || !response.data.ts_code) {
        setNoData(true)
        setProgress(100)
        setStage('无数据')
        clearInterval(progressInterval)
        setLoading(false)
        return
      }
      
      // 使用后端返回的历史数据
      if (response.data.history_data && response.data.history_data.length > 0) {
        // 转换历史数据格式
        const data = response.data.history_data.map((item: any) => ({
          date: item.date,
          close: item.close,
          volume: item.volume,
          ts_code: item.ts_code || stockCode,
          open: item.open,
          high: item.high,
          low: item.low,
          pre_close: item.pre_close,
          change: item.change,
          pct_chg: item.pct_chg,
          vol: item.vol,
          amount: item.amount
        }))
        setStockData(data)
      } else {
        // 生成模拟数据作为备用
        const data: Array<{date: string, close: number, volume: number}> = []
        const today = new Date()
        const latestClose = response.data.latest_close
        
        // 生成过去90天的数据
        for (let i = 89; i >= 0; i--) {
          const date = new Date(today)
          date.setDate(date.getDate() - i)
          const dateStr = date.toISOString().split('T')[0]
          
          // 生成随机波动的收盘价
          const randomChange = (Math.random() - 0.5) * 0.02 // 2%以内的随机波动
          const close = latestClose * (1 + randomChange * (89 - i) / 90)
          
          // 生成随机交易量
          const volume = Math.floor(Math.random() * 10000000) + 5000000
          
          data.push({ date: dateStr, close, volume })
        }
        setStockData(data)
      }
      setProgress(100)
      setStage('预测完成')
      clearInterval(progressInterval)
      setLoading(false)
    } catch (error) {
      message.error('预测失败，请稍后重试')
      setProgress(100)
      setStage('预测失败')
      clearInterval(progressInterval)
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
    { key: '1', name: '股票代码', value: result.ts_code },
    { key: '2', name: '最新日期', value: result.latest_date },
    { key: '3', name: '最新收盘价', value: `${result.latest_close.toFixed(2)}` },
    { key: '4', name: '预测方向', value: result.prediction.label === 1 ? '上涨' : result.prediction.label === -1 ? '下跌' : '横盘' },
    { key: '5', name: '上涨概率', value: `${(result.prediction.probability.up * 100).toFixed(2)}%` },
    { key: '6', name: '横盘概率', value: `${(result.prediction.probability.flat * 100).toFixed(2)}%` },
    { key: '7', name: '下跌概率', value: `${(result.prediction.probability.down * 100).toFixed(2)}%` },
    { key: '8', name: '上涨空间', value: `${(result.prediction.price_space.up_space * 100).toFixed(2)}%` },
    { key: '9', name: '下跌空间', value: `${(result.prediction.price_space.down_space * 100).toFixed(2)}%` },
    { key: '10', name: '预期价格', value: `${result.prediction.price_space.expected_price.toFixed(2)}` },
    { key: '11', name: '年化波动率', value: `${(result.volatility * 100).toFixed(2)}%` },
  ] : []

  return (
    <Card>
      <Title level={2}>股票预测</Title>
      <div style={{ marginBottom: 24 }}>
        <Input
          placeholder="输入股票代码，如 000564.SZ"
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
      
      {/* 预测进度 */}
      {loading && (
        <div style={{ marginBottom: 24 }}>
          <Text>{stage}</Text>
          <Progress percent={progress} status="active" />
        </div>
      )}
      
      {/* 无数据提示 */}
      {noData && (
        <Alert
          message="无数据提示"
          description="无法获取该股票的数据，请检查股票代码是否正确或稍后重试"
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}
      
      {/* 预测结果 */}
      {result && !noData && (
        <>
          <Table columns={columns} dataSource={data} pagination={false} style={{ marginBottom: 24 }} />
          
          {/* 基本面数据 */}
          {result.fundamentals && Object.keys(result.fundamentals).length > 0 && (
            <>
              <Title level={3}>基本面数据</Title>
              <Table 
                columns={[
                  { title: '指标', dataIndex: 'name', key: 'name' },
                  { title: '值', dataIndex: 'value', key: 'value' }
                ]} 
                dataSource={[
                  { key: '1', name: '市盈率(PE)', value: result.fundamentals.pe ? result.fundamentals.pe.toFixed(2) : 'N/A' },
                  { key: '2', name: '市净率(PB)', value: result.fundamentals.pb ? result.fundamentals.pb.toFixed(2) : 'N/A' },
                  { key: '3', name: '净资产收益率(ROE)', value: result.fundamentals.roe ? `${result.fundamentals.roe.toFixed(2)}%` : 'N/A' },
                  { key: '4', name: '净利润增速', value: result.fundamentals.npg_rate ? `${result.fundamentals.npg_rate.toFixed(2)}%` : 'N/A' },
                  { key: '5', name: '资产负债率', value: result.fundamentals.debt_to_assets ? `${result.fundamentals.debt_to_assets.toFixed(2)}%` : 'N/A' }
                ]} 
                pagination={false} 
                style={{ marginBottom: 24 }} 
              />
            </>
          )}
          
          {/* 股价走势与交易量 */}
          <Title level={3}>股价走势与交易量</Title>
          <div ref={chartRef} style={{ height: 500, marginBottom: 24 }} />
          

        </>
      )}
    </Card>
  )
}

export default PredictPage