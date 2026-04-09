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
  trading_signals: {
    buy_signal: boolean
    sell_signal: boolean
    buy_probability: number
    sell_probability: number
    message: string
  }
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
    buy_signal: boolean
    sell_signal: boolean
    buy_price: number
    sell_price: number
    pred_golden_cross: boolean
    pred_golden_cross_proba: number
    pred_death_cross: boolean
    pred_death_cross_proba: number
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
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<{ xgboost: PredictionResult | null, lightgbm: PredictionResult | null }>({ xgboost: null, lightgbm: null })
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

      // 准备买入点和卖出点数据
        const buyPoints = [];
        const sellPoints = [];
        const predBuyPoints = [];
        const predSellPoints = [];
        const modelBuyPoints = [];  // 基于模型预测的买入点
        const predGoldenCrossPoints = [];
        const predDeathCrossPoints = [];
        const actualGoldenCrossPoints = [];
        const actualDeathCrossPoints = [];
        
        for (let i = 0; i < stockData.length; i++) {
          const item = stockData[i];
          if (item.buy_signal && item.buy_price > 0) {
            buyPoints.push([i, item.buy_price]);
          }
          if (item.sell_signal && item.sell_price > 0) {
            sellPoints.push([i, item.sell_price]);
          }
          // 预测买入点和卖出点（基于t-1数据预测）
          if (item.buy_price > 0) {
            predBuyPoints.push([i, item.buy_price]);
            // 区分基于模型的买入点：连续下降天数>0且预测收盘价>0
            if (item.consecutive_down > 0 && item.pred_close > 0) {
              modelBuyPoints.push([i, item.buy_price]);
            }
          }
          if (item.sell_price > 0) {
            predSellPoints.push([i, item.sell_price]);
          }
          if (item.pred_golden_cross) {
            predGoldenCrossPoints.push([i, item.close]);
          }
          if (item.pred_death_cross) {
            predDeathCrossPoints.push([i, item.close]);
          }
          if (item.golden_cross === 1) {
            actualGoldenCrossPoints.push([i, item.close]);
          }
          if (item.death_cross === 1) {
            actualDeathCrossPoints.push([i, item.close]);
          }
        }

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
              tooltipContent += `<div>开盘价: ${data.open !== undefined && data.open !== null ? data.open.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>最高价: ${data.high !== undefined && data.high !== null ? data.high.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>最低价: ${data.low !== undefined && data.low !== null ? data.low.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>收盘价: ${data.close.toFixed(2)}</div>`;
              tooltipContent += `<div>昨收价: ${data.pre_close !== undefined && data.pre_close !== null ? data.pre_close.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>涨跌额: ${data.change !== undefined && data.change !== null ? data.change.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>涨跌幅: ${data.pct_chg !== undefined && data.pct_chg !== null ? data.pct_chg.toFixed(2) : 'N/A'}%</div>`;
              tooltipContent += `<div>成交量: ${data.vol !== undefined && data.vol !== null ? data.vol.toFixed(2) : 'N/A'} </div>`;
              tooltipContent += `<div>成交额: ${data.amount !== undefined && data.amount !== null ? data.amount.toFixed(2) : 'N/A'} 千元</div>`;
              // 交易信号
              tooltipContent += `<div style="margin-top:5px; font-weight:bold">交易信号:</div>`;
              tooltipContent += `<div>买入信号: ${data.buy_signal ? '是' : '否'}</div>`;
              tooltipContent += `<div>卖出信号: ${data.sell_signal ? '是' : '否'}</div>`;
              if (data.buy_signal && data.buy_price > 0) {
                tooltipContent += `<div>买入价格: ${data.buy_price.toFixed(2)}</div>`;
              }
              if (data.sell_signal && data.sell_price > 0) {
                tooltipContent += `<div>卖出价格: ${data.sell_price.toFixed(2)}</div>`;
              }
              // 金叉和死叉信息
              tooltipContent += `<div style="margin-top:5px; font-weight:bold">金叉/死叉:</div>`;
              tooltipContent += `<div>预测金叉: ${data.pred_golden_cross ? '是' : '否'} (${(data.pred_golden_cross_proba * 100).toFixed(2)}%)</div>`;
              tooltipContent += `<div>预测死叉: ${data.pred_death_cross ? '是' : '否'} (${(data.pred_death_cross_proba * 100).toFixed(2)}%)</div>`;
              tooltipContent += `<div>实际金叉: ${data.golden_cross === 1 ? '是' : '否'}</div>`;
              tooltipContent += `<div>实际死叉: ${data.death_cross === 1 ? '是' : '否'}</div>`;
              // 预测价格和买卖点
              tooltipContent += `<div style="margin-top:5px; font-weight:bold">预测买卖点 (基于t-1数据):</div>`;
              tooltipContent += `<div>预测收盘价: ${data.pred_close !== undefined && data.pred_close > 0 ? data.pred_close.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>预测买入点: ${data.buy_price !== undefined && data.buy_price > 0 ? data.buy_price.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>预测卖出点: ${data.sell_price !== undefined && data.sell_price > 0 ? data.sell_price.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>ATR (平均真实波幅): ${data.atr !== undefined && data.atr > 0 ? data.atr.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>连续下降天数: ${data.consecutive_down !== undefined && data.consecutive_down > 0 ? data.consecutive_down : '0'}</div>`;
              
              // 模型买入点标识
              if (data.consecutive_down > 0 && data.pred_close > 0 && data.buy_price > 0) {
                tooltipContent += `<div style="margin-top:5px; color:#1890ff; font-weight:bold">★ 模型买入点</div>`;
                tooltipContent += `<div style="color:#1890ff">基于连续下降${data.consecutive_down}天预测</div>`;
              }
              
              // 技术指标
              tooltipContent += `<div style="margin-top:5px; font-weight:bold">技术指标:</div>`;
              tooltipContent += `<div>MA5 (5日均线): ${data.ma5 !== undefined && data.ma5 !== null ? data.ma5.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>MA10 (10日均线): ${data.ma10 !== undefined && data.ma10 !== null ? data.ma10.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>MA20 (20日均线): ${data.ma20 !== undefined && data.ma20 !== null ? data.ma20.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>MA60 (60日均线): ${data.ma60 !== undefined && data.ma60 !== null ? data.ma60.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>EMA12 (12日指数移动平均线): ${data.ema12 !== undefined && data.ema12 !== null ? data.ema12.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>EMA26 (26日指数移动平均线): ${data.ema26 !== undefined && data.ema26 !== null ? data.ema26.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>RSI (相对强弱指数): ${data.rsi !== undefined && data.rsi !== null ? data.rsi.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>DIF (差离值): ${data.macd !== undefined && data.macd !== null ? data.macd.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>DEA (信号线): ${data.macd_signal !== undefined && data.macd_signal !== null ? data.macd_signal.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>MACD Hist (MACD柱状图): ${data.macd_hist !== undefined && data.macd_hist !== null ? data.macd_hist.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>布林上轨 (布林带上限): ${data.boll_upper !== undefined && data.boll_upper !== null ? data.boll_upper.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>布林中轨 (20日均线): ${data.boll_mid !== undefined && data.boll_mid !== null ? data.boll_mid.toFixed(2) : 'N/A'}</div>`;
              tooltipContent += `<div>布林下轨 (布林带下限): ${data.boll_lower !== undefined && data.boll_lower !== null ? data.boll_lower.toFixed(2) : 'N/A'}</div>`;
              return tooltipContent;
            }
          },
          legend: {
            data: ['收盘价', '交易量', '买入点', '卖出点', '预测买入点', '模型买入点', '预测卖出点', '预测金叉', '预测死叉', '实际金叉', '实际死叉']
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
            },
            {
              name: '买入点',
              type: 'scatter',
              data: buyPoints,
              yAxisIndex: 0,
              symbolSize: 8,
              itemStyle: {
                color: '#52c41a'
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '卖出点',
              type: 'scatter',
              data: sellPoints,
              yAxisIndex: 0,
              symbolSize: 8,
              itemStyle: {
                color: '#ff4d4f'
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '预测金叉',
              type: 'scatter',
              data: predGoldenCrossPoints,
              yAxisIndex: 0,
              symbol: 'triangle',
              symbolSize: 10,
              itemStyle: {
                color: '#1890ff',
                borderColor: '#fff',
                borderWidth: 2
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '预测死叉',
              type: 'scatter',
              data: predDeathCrossPoints,
              yAxisIndex: 0,
              symbol: 'triangle',
              symbolSize: 10,
              symbolRotate: 180,
              itemStyle: {
                color: '#faad14',
                borderColor: '#fff',
                borderWidth: 2
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '实际金叉',
              type: 'scatter',
              data: actualGoldenCrossPoints,
              yAxisIndex: 0,
              symbol: 'circle',
              symbolSize: 8,
              itemStyle: {
                color: '#52c41a',
                borderColor: '#fff',
                borderWidth: 2
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '实际死叉',
              type: 'scatter',
              data: actualDeathCrossPoints,
              yAxisIndex: 0,
              symbol: 'circle',
              symbolSize: 8,
              itemStyle: {
                color: '#ff4d4f',
                borderColor: '#fff',
                borderWidth: 2
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '预测买入点',
              type: 'scatter',
              data: predBuyPoints,
              yAxisIndex: 0,
              symbol: 'circle',
              symbolSize: 6,
              itemStyle: {
                color: '#52c41a',
                borderColor: '#fff',
                borderWidth: 1
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '模型买入点',
              type: 'scatter',
              data: modelBuyPoints,
              yAxisIndex: 0,
              symbol: 'diamond',
              symbolSize: 10,
              itemStyle: {
                color: '#1890ff',
                borderColor: '#fff',
                borderWidth: 2
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
              }
            },
            {
              name: '预测卖出点',
              type: 'scatter',
              data: predSellPoints,
              yAxisIndex: 0,
              symbol: 'circle',
              symbolSize: 6,
              itemStyle: {
                color: '#ff4d4f',
                borderColor: '#fff',
                borderWidth: 1
              },
              emphasis: {
                itemStyle: {
                  shadowBlur: 10,
                  shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
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
      if (chartInstance.current) {
        chartInstance.current.dispose()
        chartInstance.current = null
      }
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
      // 同时调用两个模型的预测
      setStage('获取股票数据')
      const [xgboostResponse, lightgbmResponse] = await Promise.all([
        axios.post('http://localhost:5001/api/predict', { ts_code: stockCode, model_type: 'xgboost' }),
        axios.post('http://localhost:5001/api/predict', { ts_code: stockCode, model_type: 'lightgbm' })
      ])

      setStage('分析数据')
      setProgress(70)

      setStage('生成预测结果')
      setResults({
        xgboost: xgboostResponse.data,
        lightgbm: lightgbmResponse.data
      })

      // 检查是否有数据
      if (!xgboostResponse.data || !xgboostResponse.data.ts_code) {
        setNoData(true)
        setProgress(100)
        setStage('无数据')
        clearInterval(progressInterval)
        setLoading(false)
        return
      }

      // 使用后端返回的历史数据（使用xgboost的历史数据）
      if (xgboostResponse.data.history_data && xgboostResponse.data.history_data.length > 0) {
        // 转换历史数据格式
        const data = xgboostResponse.data.history_data.map((item: any) => ({
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
          vol: item.volume,  // 使用volume字段作为vol的值
          amount: item.amount,
          ma5: item.ma5,  // SMA_5
          ma10: item.ma10,
          ma20: item.ma20,  // SMA_20
          ema12: item.ema12,  // EMA_12
          ema26: item.ema26,  // EMA_26
          rsi: item.rsi,
          macd: item.macd,  // DIF
          macd_signal: item.macd_signal,  // DEA
          macd_hist: item.macd_hist,  // MACD Hist
          ma60: item.ma60,  // 60日均线
          atr: item.atr,
          golden_cross: item.golden_cross,
          death_cross: item.death_cross,
          buy_signal: item.buy_signal,
          sell_signal: item.sell_signal,
          buy_price: item.buy_price,
          sell_price: item.sell_price,
          pred_golden_cross: item.pred_golden_cross,
          pred_golden_cross_proba: item.pred_golden_cross_proba,
          pred_death_cross: item.pred_death_cross,
          pred_death_cross_proba: item.pred_death_cross_proba
        }))
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
      {results.xgboost && results.lightgbm && !noData && (
        <>
          {/* 交易信号 */}
          <Title level={3}>交易信号</Title>
          <Table
            columns={columns}
            dataSource={results.xgboost.trading_signals ? [
              { key: '1', name: '买入信号', value: results.xgboost.trading_signals.buy_signal ? '是' : '否' },
              { key: '2', name: '卖出信号', value: results.xgboost.trading_signals.sell_signal ? '是' : '否' },
              { key: '3', name: '买入概率', value: `${(results.xgboost.trading_signals.buy_probability * 100).toFixed(2)}%` },
              { key: '4', name: '卖出概率', value: `${(results.xgboost.trading_signals.sell_probability * 100).toFixed(2)}%` },
              { key: '5', name: '交易建议', value: results.xgboost.trading_signals.message },
            ] : []}
            pagination={false}
            style={{ marginBottom: 24 }}
          />
          
          {/* div flex 布局 */}
          <div style={{ display: 'flex', flexDirection: 'row',justifyContent: 'center',alignContent: 'center',gap: 24 }}>
            <div style={{flex:1}}>


              {/* XGBoost 预测结果 */}
              <Title level={3}>XGBoost 预测结果</Title>
              <Table
                columns={columns}
                dataSource={results.xgboost ? [
                  { key: '1', name: '股票代码', value: results.xgboost.ts_code },
                  { key: '2', name: '最新日期', value: results.xgboost.latest_date },
                  { key: '3', name: '最新收盘价', value: `${results.xgboost.latest_close.toFixed(2)}` },
                  { key: '4', name: '预测方向', value: results.xgboost.prediction.label === 1 ? '上涨' : results.xgboost.prediction.label === -1 ? '下跌' : '横盘' },
                  { key: '5', name: '上涨概率', value: `${(results.xgboost.prediction.probability.up * 100).toFixed(2)}%` },
                  { key: '6', name: '横盘概率', value: `${(results.xgboost.prediction.probability.flat * 100).toFixed(2)}%` },
                  { key: '7', name: '下跌概率', value: `${(results.xgboost.prediction.probability.down * 100).toFixed(2)}%` },
                  { key: '8', name: '上涨空间', value: `${(results.xgboost.prediction.price_space.up_space * 100).toFixed(2)}%` },
                  { key: '9', name: '下跌空间', value: `${(results.xgboost.prediction.price_space.down_space * 100).toFixed(2)}%` },
                  { key: '10', name: '预期价格', value: `${results.xgboost.prediction.price_space.expected_price.toFixed(2)}` },
                  { key: '11', name: '年化波动率', value: `${(results.xgboost.volatility * 100).toFixed(2)}%` },
                ] : []}
                pagination={false}
                style={{ marginBottom: 24 }}
              />

            </div>
            <div style={{flex:1}}>
              {/* LightGBM 预测结果 */}
              <Title level={3}>LightGBM 预测结果</Title>
              <Table
                columns={columns}
                dataSource={results.lightgbm ? [
                  { key: '1', name: '股票代码', value: results.lightgbm.ts_code },
                  { key: '2', name: '最新日期', value: results.lightgbm.latest_date },
                  { key: '3', name: '最新收盘价', value: `${results.lightgbm.latest_close.toFixed(2)}` },
                  { key: '4', name: '预测方向', value: results.lightgbm.prediction.label === 1 ? '上涨' : results.lightgbm.prediction.label === -1 ? '下跌' : '横盘' },
                  { key: '5', name: '上涨概率', value: `${(results.lightgbm.prediction.probability.up * 100).toFixed(2)}%` },
                  { key: '6', name: '横盘概率', value: `${(results.lightgbm.prediction.probability.flat * 100).toFixed(2)}%` },
                  { key: '7', name: '下跌概率', value: `${(results.lightgbm.prediction.probability.down * 100).toFixed(2)}%` },
                  { key: '8', name: '上涨空间', value: `${(results.lightgbm.prediction.price_space.up_space * 100).toFixed(2)}%` },
                  { key: '9', name: '下跌空间', value: `${(results.lightgbm.prediction.price_space.down_space * 100).toFixed(2)}%` },
                  { key: '10', name: '预期价格', value: `${results.lightgbm.prediction.price_space.expected_price.toFixed(2)}` },
                  { key: '11', name: '年化波动率', value: `${(results.lightgbm.volatility * 100).toFixed(2)}%` },
                ] : []}
                pagination={false}
                style={{ marginBottom: 24 }}
              />
            </div>
          </div>

          {/* 基本面数据 */}
          {results.xgboost.fundamentals && Object.keys(results.xgboost.fundamentals).length > 0 && (
            <>
              <Title level={3}>基本面数据</Title>
              <Table
                columns={[
                  { title: '指标', dataIndex: 'name', key: 'name' },
                  { title: '值', dataIndex: 'value', key: 'value' }
                ]}
                dataSource={[
                  { key: '1', name: '市盈率(PE)', value: results.xgboost.fundamentals.pe ? results.xgboost.fundamentals.pe.toFixed(2) : 'N/A' },
                  { key: '2', name: '市净率(PB)', value: results.xgboost.fundamentals.pb ? results.xgboost.fundamentals.pb.toFixed(2) : 'N/A' },
                  { key: '3', name: '净资产收益率(ROE)', value: results.xgboost.fundamentals.roe ? `${results.xgboost.fundamentals.roe.toFixed(2)}%` : 'N/A' },
                  { key: '4', name: '净利润增速', value: results.xgboost.fundamentals.npg_rate ? `${results.xgboost.fundamentals.npg_rate.toFixed(2)}%` : 'N/A' },
                  { key: '5', name: '资产负债率', value: results.xgboost.fundamentals.debt_to_assets ? `${results.xgboost.fundamentals.debt_to_assets.toFixed(2)}%` : 'N/A' }
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