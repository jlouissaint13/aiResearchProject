import { useState, useEffect, useRef } from 'react';
import { Box, Typography, TextField, IconButton, LinearProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import AssistantIcon from '@mui/icons-material/Assistant';
import DownloadIcon from '@mui/icons-material/Download';
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const LOADING_MESSAGES = [
    "Analyzing your data...",
    "Generating visualization...",
    "Preparing your chart...",
    "Almost ready...",
    "Finalizing visualization..."
];

interface ChartData {
    chart_type: 'scatter' | 'line' | 'bar';
    data: any[];
    title?: string;
    xKey?: string;
    yKey?: string;
    xLabel?: string;
    yLabel?: string;
}

function DataVisualizer() {
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentLoadingMessage, setCurrentLoadingMessage] = useState('');
    const [chartData, setChartData] = useState<ChartData | null>(null);
    const valueRef = useRef(0);
    const navigate = useNavigate();
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const chartRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (isLoading) {
            loadingMessagesControl(valueRef.current);
            intervalRef.current = setInterval(() => {
                loadingMessagesControl(valueRef.current)
            }, 5000)
        } else {
            setCurrentLoadingMessage('');
            valueRef.current = 0;
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [isLoading]);

    function loadingMessagesControl(i: number): void {
        if (i < LOADING_MESSAGES.length) {
            setCurrentLoadingMessage(LOADING_MESSAGES[i]);
            const next: number = i + 1;
            valueRef.current = next;
        }
    }

    /*
    async function sendMessage() {
        if (!input.trim()) return;
        setIsLoading(true);

        // MOCK BACKEND (using your test data)
        setTimeout(() => {
            const mockChartData = {
                chart_type: "line",
                title: "VCN vs Time Analysis",
                data: [
                    { vcn: 1.5, time_months: 3.5 },
                    { vcn: 2.1, time_months: 6.5 },
                    { vcn: 1.8, time_months: 4.2 },
                    { vcn: 2.5, time_months: 7.8 },
                    { vcn: 1.2, time_months: 2.1 },
                    { vcn: 3.0, time_months: 8.5 },
                    { vcn: 2.8, time_months: 7.2 },
                    { vcn: 1.9, time_months: 5.3 }
                ],
                xKey: "vcn",
                yKey: "time_months",
                xLabel: "VCN Value",
                yLabel: "Time (Months)"
            };
            setChartData(mockChartData);
            setIsLoading(false);
            setInput('');
        }, 2000);

    }
    */

    async function sendMessage() {
        if (!input.trim()) return;
        setIsLoading(true);

        const userMessage = {
            content: input.trim(),
            sender: 'user',
            user_id: localStorage.getItem("userID"),
            logged_in: localStorage.getItem("loggedIn"),
            mode: 'data_visual'
        };

        try {
            const response = await fetch('http://localhost:8000/message/send_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(userMessage),
            });

            if (response.ok) {
                const chart = await response.json();

                setChartData(chart);
                setInput('');
            }
        } catch (error) {
            alert(error)
        } finally {
            setIsLoading(false);
        }
    }

    const handleGoBack = () => {
        if (isLoading) return;
        try {
            navigate("/Choice");
        } catch (e) {
            console.log("Navigate failed, trying window.history");
            window.history.back();
        }
    };

    const handleDownload = () => {
        if (!chartRef.current) return;

        import('html2canvas').then((html2canvas) => {
            html2canvas.default(chartRef.current!).then((canvas) => {
                const link = document.createElement('a');
                link.download = `chart-${Date.now()}.png`;
                link.href = canvas.toDataURL();
                link.click();
            });
        }).catch(() => {
            alert("Please install html2canvas for download functionality: npm install html2canvas");
        });
    };

    const renderChart = () => {
        if (!chartData) return null;

        const { chart_type, data, title, xKey, yKey, xLabel, yLabel } = chartData;

        const detectedXKey = xKey || Object.keys(data[0] || {})[0] || 'x';
        const detectedYKey = yKey || Object.keys(data[0] || {})[1] || 'y';

        const commonProps = {
            margin: { top: 20, right: 30, bottom: 40, left: 40 }
        };

        const xAxisProps = {
            dataKey: detectedXKey,
            stroke: "#8e8e8e",
            label: { value: xLabel || detectedXKey, position: 'bottom', fill: '#e0e0e0', fontSize: 14, dy: 10 }
        };

        const yAxisProps = {
            stroke: "#8e8e8e",
            label: { value: yLabel || detectedYKey, angle: -90, position: 'left', fill: '#e0e0e0', fontSize: 14, dx: -10 }
        };

        const CustomTooltip = ({ active, payload, label }) => {
            if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                    <Box sx={{ bgcolor: '#282a2e', border: '1px solid #3e4042', borderRadius: '8px', p: 1.5 }}>
                        {Object.keys(data).map(key => (
                            <Typography key={key} sx={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                <span style={{ textTransform: 'capitalize', color: '#8e8e8e' }}>{key}: </span>{data[key]}
                            </Typography>
                        ))}
                    </Box>
                );
            }
            return null;
        };

        switch (chart_type.toLowerCase()) {
            case 'scatter':
                return (
                    <ResponsiveContainer width="100%" height={500}>
                        <ScatterChart {...commonProps}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#3e4042" />
                            <XAxis type="number" dataKey={detectedXKey} name={xLabel || detectedXKey} {...xAxisProps} />
                            <YAxis type="number" dataKey={detectedYKey} name={yLabel || detectedYKey} {...yAxisProps} />
                            <Tooltip content={<CustomTooltip />} />
                            <Scatter data={data} fill="#1a73e8" />
                        </ScatterChart>
                    </ResponsiveContainer>
                );

            case 'line':
                return (
                    <ResponsiveContainer width="100%" height={500}>
                        <LineChart data={data} {...commonProps}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#3e4042" />
                            <XAxis {...xAxisProps} />
                            <YAxis {...yAxisProps} />
                            <Tooltip content={<CustomTooltip />} />
                            <Line
                                type="monotone"
                                dataKey={detectedYKey}
                                stroke="#1a73e8"
                                strokeWidth={3}
                                dot={{ fill: '#1a73e8', r: 5 }}
                                activeDot={{ r: 7 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                );

            case 'bar':
                return (
                    <ResponsiveContainer width="100%" height={500}>
                        <BarChart data={data} {...commonProps}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#3e4042" />
                            <XAxis {...xAxisProps} />
                            <YAxis {...yAxisProps} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar dataKey={detectedYKey} fill="#1a73e8" />
                        </BarChart>
                    </ResponsiveContainer>
                );

            default:
                return (
                    <Typography sx={{ color: '#8e8e8e', textAlign: 'center', p: 2 }}>
                        Unsupported chart type: {chart_type}
                    </Typography>
                );
        }
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                height: '100vh',
                width: '100vw',
                overflow: 'hidden',
                background: 'linear-gradient(135deg, #1A2027 0%, #171A21 100%)',
                color: '#e0e0e0',
                fontFamily: 'Roboto, sans-serif',
                '& *': {
                    outline: 'none !important',
                },
                '& *:focus': {
                    outline: 'none !important',
                },
            }}
        >
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 2,
                    bgcolor: 'rgba(30, 32, 35, 0.98)',
                    borderBottom: '1px solid #3e4042',
                    boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
                    flexShrink: 0,
                    position: 'relative',
                }}
            >
                <IconButton
                    onClick={handleGoBack}
                    disabled={isLoading}
                    sx={{
                        color: '#8e8e8e',
                        '&:hover': { color: '#e0e0e0', bgcolor: 'transparent' },
                    }}
                >
                    <ExitToAppIcon />
                </IconButton>

                <Box sx={{ display: 'flex', alignItems: 'center', position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
                    <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                        Marie
                    </Typography>
                    <AssistantIcon sx={{ color: '#1a73e8', ml: 1.5 }} />
                </Box>

                {chartData && (
                    <IconButton
                        onClick={handleDownload}
                        sx={{
                            color: '#1a73e8',
                            '&:hover': { color: '#1565c0', bgcolor: 'transparent' },
                        }}
                    >
                        <DownloadIcon />
                    </IconButton>
                )}

                {isLoading && (
                    <LinearProgress
                        sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            height: '4px',
                            bgcolor: 'transparent',
                            '& .MuiLinearProgress-bar': {
                                bgcolor: '#1a73e8'
                            }
                        }}
                    />
                )}
            </Box>

            <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 3, display: 'flex', flexDirection: 'column', gap: 3 }}>

                {!chartData && !isLoading ? (
                    <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <Typography variant="h5" sx={{ color: '#8e8e8e' }}>
                            Describe the visualization you'd like to create
                        </Typography>
                    </Box>
                ) : isLoading ? (
                    <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2 }}>
                        <Typography variant="h6" sx={{ color: '#8e8e8e', fontStyle: 'italic' }}>
                            {currentLoadingMessage}
                        </Typography>
                    </Box>
                ) : (
                    <Box
                        ref={chartRef}
                        sx={{
                            flexGrow: 1,
                            bgcolor: '#282a2e',
                            borderRadius: 2,
                            p: 3,
                            boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.4)',
                            display: 'flex',
                            flexDirection: 'column',
                            minHeight: 500,
                        }}
                    >
                        {chartData?.title && (
                            <Typography
                                variant="h5"
                                sx={{
                                    color: '#e0e0e0',
                                    mb: 3,
                                    textAlign: 'center',
                                    fontWeight: 600
                                }}
                            >
                                {chartData.title}
                            </Typography>
                        )}

                        {renderChart()}
                    </Box>
                )}
            </Box>

            <Box sx={{
                p: 2,
                bgcolor: 'rgba(30, 32, 35, 0.98)',
                borderTop: '1px solid #3e4042',
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                flexShrink: 0,
                boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.3)'
            }}>
                <TextField
                    fullWidth
                    multiline
                    maxRows={5}
                    variant="outlined"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    }}
                    placeholder="Describe your data visualization (Only bar, line and, scatter visualizations are supported)"
                    disabled={isLoading}
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 4,
                            bgcolor: '#282a2e',
                            '& fieldset': { borderColor: '#3e4042' },
                            '&:hover fieldset': { borderColor: '#5e6062' },
                            '&.Mui-focused fieldset': { borderColor: '#1a73e8', borderWidth: '2px' },
                        },
                        '& .MuiInputBase-input': {
                            color: '#e0e0e0',
                            '::-webkit-scrollbar': { display: 'none' },
                            msOverflowStyle: 'none',
                            scrollbarWidth: 'none'
                        },
                    }}
                />
                <IconButton
                    color="primary"
                    onClick={sendMessage}
                    disabled={!input.trim() || isLoading}
                    sx={{
                        p: 1.5,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        transition: 'background-color 0.3s, box-shadow 0.1s, transform 0.1s',
                        '&:hover': {
                            bgcolor: '#1565c0',
                            transform: 'translateY(-1px)',
                            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.4)',
                        },
                        '&.Mui-disabled': { bgcolor: '#282a2e', color: '#8e8e8e' },
                    }}
                >
                    <SendIcon />
                </IconButton>
            </Box>
        </Box>
    );
}

export default DataVisualizer;