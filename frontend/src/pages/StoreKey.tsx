import { useState } from 'react';
import { Box, Typography, TextField, Button, Alert } from '@mui/material';
import KeyIcon from '@mui/icons-material/Key';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import {useNavigate} from "react-router-dom";

const ApiKeyInput = () => {
    const [apiKey, setApiKey] = useState('');
    const [isSaved, setIsSaved] = useState(false);

    const navigate = useNavigate()
    const handleSave = () => {
        if (apiKey.trim()) {
            alert("API Key was saved")
            setIsSaved(true);

            setTimeout(() => setIsSaved(false), 3000);
        }
    };

    const handleBack = () => {
        navigate("/Choice")
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'absolute',
                inset: 0,
                background: 'linear-gradient(135deg, #1A2027 0%, #171A21 100%)',
                color: '#e0e0e0',
                fontFamily: 'Roboto, sans-serif',
                p: 3,
            }}
        >
            <Box
                sx={{
                    position: 'absolute',
                    top: { xs: 16, md: 20 },
                    left: { xs: 16, md: 20 },
                    zIndex: 10,
                }}
            >
                <Button
                    onClick={handleBack}
                    variant="text"
                    startIcon={<ArrowBackIcon />}
                    sx={{
                        color: '#8e8e8e',
                        textTransform: 'none',
                        fontSize: '0.85rem',
                        p: 0.5,
                        borderRadius: 1,
                        '&:hover': {
                            bgcolor: 'rgba(255, 255, 255, 0.05)',
                            color: '#e0e0e0',
                        },
                    }}
                >
                    Back to Menu
                </Button>
            </Box>

            <Box
                sx={{
                    p: { xs: 4, md: 5 },
                    bgcolor: 'rgba(30, 32, 35, 0.98)',
                    backdropFilter: 'blur(8px)',
                    borderRadius: 3,
                    boxShadow: '0 8px 30px rgba(0,0,0,0.7)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 3,
                    width: '100%',
                    maxWidth: 500,
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                }}
            >
                <KeyIcon sx={{ fontSize: 60, color: '#1a73e8' }} />

                <Typography
                    variant="h5"
                    component="h1"
                    sx={{
                        color: '#e0e0e0',
                        fontWeight: 600,
                        letterSpacing: 0.5,
                        textTransform: 'uppercase',
                    }}
                >
                    Enter API Key
                </Typography>
                <Typography variant="body2" sx={{ color: '#8e8e8e', textAlign: 'center', mt: -1 }}>
                    Paste your key to enable external model access.
                </Typography>

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Enter your API Key here"
                    value={apiKey}
                    onChange={(e) => {
                        setApiKey(e.target.value);
                        setIsSaved(false);
                    }}
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 1,
                            bgcolor: '#282a2e',
                            '& fieldset': {
                                borderColor: '#3e4042',
                                transition: 'border-color 0.3s',
                            },
                            '&:hover fieldset': {
                                borderColor: '#5e6062',
                            },
                            '&.Mui-focused fieldset': {
                                borderColor: '#1a73e8',
                                borderWidth: '2px',
                            },
                        },
                        '& .MuiInputBase-input': { color: '#e0e0e0', letterSpacing: '1px' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                        mt: 1,
                    }}
                />

                <Button
                    fullWidth
                    variant="contained"
                    onClick={handleSave}
                    disabled={!apiKey.trim()}
                    sx={{
                        p: 1.25,
                        borderRadius: 1,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        textTransform: 'none',
                        fontWeight: 500,
                        transition: 'background-color 0.3s, box-shadow 0.1s',
                        '&:hover': {
                            bgcolor: '#1565c0',
                            transform: 'translateY(-1px)',
                            boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)',
                        },
                        '&.Mui-disabled': {
                            bgcolor: '#282a2e',
                            color: '#8e8e8e',
                            border: '1px solid #3e4042'
                        }
                    }}
                >
                    Save Key
                </Button>

                {isSaved && (
                    <Alert
                        severity="success"
                        icon={false}
                        sx={{
                            width: '100%',
                            mt: 1,
                            borderRadius: 1,
                            bgcolor: '#282a2e',
                            color: '#4caf50',
                            border: '1px solid #4caf50',
                            justifyContent: 'center'
                        }}
                    >
                        API Key successfully saved!
                    </Alert>
                )}
            </Box>
        </Box>
    );
};

export default ApiKeyInput;