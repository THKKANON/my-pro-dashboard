import React, { useState, useEffect, useMemo, memo, useRef } from 'react';
import * as echarts from 'echarts/core';
import { CandlestickChart, LineChart } from 'echarts/charts';
import { TitleComponent, TooltipComponent, GridComponent, LegendComponent, MarkPointComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import './App.css';

// ECharts에 필요한 컴포넌트들을 등록
echarts.use([
    TitleComponent, TooltipComponent, GridComponent, LegendComponent, MarkPointComponent,
    CandlestickChart, LineChart, CanvasRenderer
]);

const COIN_LIST = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'TRXUSDT', 'ETCUSDT', 'BCHUSDT'];
const TIMEFRAME_OPTIONS = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d'];
const DEFAULT_TIMEFRAMES = ['5m', '15m', '30m', '1h'];
const CANDLE_COUNT_OPTIONS = [15, 30, 50, 100, 150];

const ChartComponent = memo(({ data, title }) => {
    const chartRef = useRef(null);
    const chartInstanceRef = useRef(null);

    // 1. 차트 생성/소멸 로직
    useEffect(() => {
        if (chartRef.current) {
            chartInstanceRef.current = echarts.init(chartRef.current);
            const handleResize = () => chartInstanceRef.current?.resize();
            window.addEventListener('resize', handleResize);

            return () => {
                window.removeEventListener('resize', handleResize);
                chartInstanceRef.current?.dispose();
            };
        }
    }, []);

    // 2. 데이터 업데이트 로직
    useEffect(() => {
        if (chartInstanceRef.current) {
            if (data && data.candles && data.candles.length > 0) {
                const option = {
                    title: { text: title, left: 'center', textStyle: { color: '#d1d4dc', fontSize: 14, fontWeight: 'bold' } },
                    backgroundColor: 'transparent',
                    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
                    legend: { data: ['StochK', 'StochD', 'KernelRSI'], top: '12%', textStyle: { color: '#d1d4dc' } },
                    grid: [
                        { left: '5%', right: '5%', top: '18%', height: '42%', containLabel: true },
                        { left: '5%', right: '5%', bottom: '3%', height: '30%', containLabel: true }
                    ],
                    xAxis: [
                        { type: 'category', data: data.candles.map(d => new Date(d.Date * 1000).toLocaleTimeString('en-GB')), axisLine: { lineStyle: { color: '#8392A5' } }, axisLabel: { show: false } },
                        { type: 'category', gridIndex: 1, data: data.candles.map(d => new Date(d.Date * 1000).toLocaleTimeString('en-GB')), axisLine: { lineStyle: { color: '#8392A5' } }, axisLabel: { rotate: 45, fontSize: 10 } }
                    ],
                    yAxis: [
                        { scale: true, axisLine: { lineStyle: { color: '#8392A5' } }, splitLine: { lineStyle: { color: '#2a2e39' } }, axisLabel: { formatter: value => value.toFixed(4) } },
                        { scale: true, gridIndex: 1, axisLine: { lineStyle: { color: '#8392A5' } }, splitLine: { lineStyle: { color: '#2a2e39' } }, min: -10, max: 110 }
                    ],
                    series: [
                        { name: 'Candlestick', type: 'candlestick', data: data.candles.map(d => [d.Open, d.Close, d.Low, d.High]), itemStyle: { color: '#26A69A', color0: '#EF5350', borderColor: '#26A69A', borderColor0: '#EF5350' },
                          markPoint: { symbolSize: 16, data: [...data.stoch_buy_markers, ...data.stoch_sell_markers, ...data.krsi_long_markers, ...data.krsi_short_markers] }},
                        { name: 'StochK', type: 'line', data: data.stoch_rsi.map(d => d.STOCHRSIk_14_14_3_3), xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, lineStyle: { color: 'blue', width: 1 } },
                        { name: 'StochD', type: 'line', data: data.stoch_rsi.map(d => d.STOCHRSId_14_14_3_3), xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, lineStyle: { color: 'orange', width: 1 } },
                        { name: 'KernelRSI', type: 'line', data: data.kernel_rsi.map(d => d.Value), xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, lineStyle: { color: 'purple', width: 2 } }
                    ]
                };
                chartInstanceRef.current.setOption(option, { notMerge: false });
            } else if (chartInstanceRef.current) {
                chartInstanceRef.current.clear();
            }
        }
    }, [data, title]);

    return <div ref={chartRef} style={{ width: '100%', height: '100%' }} />;
});

function App() {
    const getInitialState = (key, defaultValue, list = []) => {
        try {
            const item = localStorage.getItem(key);
            if (key === 'selectedTfs') { return item ? item.split(',') : defaultValue; }
            if (key === 'candleCount') { return item ? parseInt(item, 10) : defaultValue; }
            return item && list.includes(item) ? item : defaultValue;
        } catch (error) { console.error("Error reading from localStorage", error); return defaultValue; }
    };
    
    const [selectedCoin, setSelectedCoin] = useState(() => getInitialState('selectedCoin', COIN_LIST[0], COIN_LIST));
    const [selectedTfs, setSelectedTfs] = useState(() => getInitialState('selectedTfs', DEFAULT_TIMEFRAMES));
    const [allChartData, setAllChartData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [candleCount, setCandleCount] = useState(() => getInitialState('candleCount', 150));

    useEffect(() => { localStorage.setItem('selectedCoin', selectedCoin); }, [selectedCoin]);
    useEffect(() => { localStorage.setItem('selectedTfs', selectedTfs.join(',')); }, [selectedTfs]);
    useEffect(() => { localStorage.setItem('candleCount', candleCount); }, [candleCount]);

    // ✨✨✨ 핵심 수정: 로딩 로직과 인터벌 로직 분리 ✨✨✨
    useEffect(() => {
        const loadDataOnCoinChange = async () => {
            setIsLoading(true); // 코인이 변경될 때만 로딩 화면 표시
            try {
                const response = await fetch(`http://127.0.0.1:8000/api/chart-data/${selectedCoin}/`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                setAllChartData(data);
            } catch (error) {
                console.error("Failed to fetch chart data:", error);
                setAllChartData({});
            } finally {
                setIsLoading(false);
            }
        };
        loadDataOnCoinChange();
    }, [selectedCoin]); // 이 useEffect는 selectedCoin이 바뀔 때만 실행됨

    useEffect(() => {
        // 백그라운드 업데이트는 로딩 화면을 표시하지 않음
        const backgroundFetch = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:8000/api/chart-data/${selectedCoin}/`);
                if (!response.ok) return;
                const data = await response.json();
                setAllChartData(data);
            } catch (error) {
                console.error("Background fetch failed:", error);
            }
        };

        const interval = setInterval(backgroundFetch, 1000);
        return () => clearInterval(interval);
    }, [selectedCoin]); // 코인이 바뀌면 기존 인터벌은 정리되고 새 인터벌이 시작됨
    
    const handleTfChange = (index, value) => { const newTfs = [...selectedTfs]; newTfs[index] = value; setSelectedTfs(newTfs); };
    
    const processedChartData = useMemo(() => {
        if (!allChartData) return [];
        return selectedTfs.map(tf => {
            if (!allChartData[tf]) return null;
            const originalData = allChartData[tf], count = parseInt(candleCount, 10);
            const slicedCandles = originalData.candles.slice(-count);
            if (slicedCandles.length === 0) return null;
            const candleTimeMap = new Map(slicedCandles.map((c, i) => [c.Date, i]));
            const generateMarkPoints = (signalTimestamps, signalType, color) => {
                const upArrowPath = 'path://M 0 -7 L 6 2 L -6 2 Z', downArrowPath = 'path://M 0 7 L 6 -2 L -6 -2 Z', starPath = 'path://M 0 -6 L 1.8 -1.8 L 6 0 L 1.8 1.8 L 0 6 L -1.8 1.8 L -6 0 L -1.8 -1.8 Z';
                let symbolPath = 'pin', yAxisCoord, symbolOffset;
                if (signalType.includes('buy')) { symbolPath = upArrowPath; yAxisCoord = 'low'; symbolOffset = [0, 10]; } 
                else if (signalType.includes('sell')) { symbolPath = downArrowPath; yAxisCoord = 'high'; symbolOffset = [0, -10]; } 
                else if (signalType.includes('long')) { symbolPath = starPath; yAxisCoord = 'low'; symbolOffset = [0, 12]; } 
                else { symbolPath = starPath; yAxisCoord = 'high'; symbolOffset = [0, -12]; }
                return signalTimestamps.map(ts => {
                    if (candleTimeMap.has(ts)) {
                        const candleIndex = candleTimeMap.get(ts), candle = slicedCandles[candleIndex];
                        return { xAxis: candleIndex, yAxis: candle[yAxisCoord === 'low' ? 'Low' : 'High'], symbol: symbolPath, symbolOffset: symbolOffset, itemStyle: { color } };
                    } return null;
                }).filter(p => p);
            };
            return {
                candles: slicedCandles,
                stoch_rsi: originalData.stoch_rsi.slice(-count), kernel_rsi: originalData.kernel_rsi.slice(-count),
                stoch_buy_markers: generateMarkPoints(originalData.stoch_buy, 'buy', '#00BFFF'),
                stoch_sell_markers: generateMarkPoints(originalData.stoch_sell, 'sell', '#FF69B4'),
                krsi_long_markers: generateMarkPoints(originalData.krsi_long, 'long', '#39FF14'),
                krsi_short_markers: generateMarkPoints(originalData.krsi_short, 'short', '#FF2400'),
            };
        });
    }, [allChartData, candleCount, selectedTfs]);

    return (
        <div className="app-container">
            <h1>{selectedCoin} Real-Time Dashboard</h1>
            <div className="controls">
                <span>Coin:</span>
                <select value={selectedCoin} onChange={e => setSelectedCoin(e.target.value)}>
                    {COIN_LIST.map(coin => <option key={coin} value={coin}>{coin}</option>)}
                </select>
                {selectedTfs.map((tf, i) => (
                    <React.Fragment key={i}>
                        <span>{`TF${i + 1}:`}</span>
                        <select value={tf} onChange={e => handleTfChange(i, e.target.value)}>
                            {TIMEFRAME_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                        </select>
                    </React.Fragment>
                ))}
                <span>Candles:</span>
                <select value={candleCount} onChange={e => setCandleCount(e.target.value)}>
                    {CANDLE_COUNT_OPTIONS.map(count => <option key={count} value={count}>{count}</option>)}
                </select>
            </div>
            {isLoading ? (<div className="loading-text">Loading initial data...</div>) : (
                <div className="chart-grid">
                    {processedChartData.map((chartData, i) => (
                        <div className="chart-wrapper" key={`${selectedCoin}-${selectedTfs[i]}`}>
                            <ChartComponent data={chartData} title={`TF${i + 1}: ${selectedTfs[i]}`} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default App;
