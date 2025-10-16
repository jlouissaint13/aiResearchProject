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
                    top: 24,
                    left: 24,
                    zIndex: 10,
                }}
            >
                <Button
                    onClick={handleBack}
                    variant="text"
                    startIcon={<ArrowBackIcon />}
                    sx={{
                        color: '#e0e0e0',
                        textTransform: 'none',
                        '&:hover': {
                            bgcolor: 'rgba(255, 255, 255, 0.08)',
                        },
                    }}
                >
                    Back to Menu
                </Button>
            </Box>

            <Box
                sx={{
                    p: { xs: 3, md: 5 },
                    bgcolor: 'rgba(41, 43, 46, 0.8)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: 4,
                    boxShadow: '0px 8px 30px rgba(0, 0, 0, 0.6)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 3,
                    width: '100%',
                    maxWidth: 500,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
            >
                <KeyIcon sx={{ fontSize: 60, color: '#1a73e8' }} />

                <Typography variant="h5" component="h1" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>
                    Enter API Key
                </Typography>
                <Typography variant="body2" sx={{ color: '#8e8e8e', textAlign: 'center' }}>
                    Paste your key to enable external model access. It will be stored securely.
                </Typography>

                <TextField
                    fullWidth
                    variant="filled"
                    label="Enter your API Key here"
                    value={apiKey}
                    onChange={(e) => {
                        setApiKey(e.target.value);
                        setIsSaved(false);
                    }}
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 2,
                            bgcolor: '#3e4042',
                            '& fieldset': { borderColor: 'transparent' },
                            '&:hover fieldset': { borderColor: '#5e5e5e' },
                            '&.Mui-focused fieldset': { borderColor: '#1a73e8' },
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
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        textTransform: 'none',
                        '&:hover': {
                            bgcolor: '#1565c0',
                            boxShadow: '0px 4px 15px rgba(26, 115, 232, 0.4)',
                        },
                    }}
                >
                    Save Key
                </Button>

                {isSaved && (
                    <Alert
                        severity="success"
                        sx={{
                            width: '100%',
                            mt: 1,
                            borderRadius: 2,
                            bgcolor: 'rgba(76, 175, 80, 0.15)',
                            color: '#4caf50',
                            border: '1px solid #4caf50',
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