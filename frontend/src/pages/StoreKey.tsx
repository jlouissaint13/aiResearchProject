import { useState } from 'react';
import {
    Box,
    Typography,
    TextField,
    Button,
    Alert,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
} from '@mui/material';
import KeyIcon from '@mui/icons-material/Key';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from "react-router-dom";
import axios from "axios";

const modalTextFieldStyles = {
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
    '& .MuiInputBase-input': { color: '#e0e0e0' },
    '& .MuiInputLabel-root': { color: '#8e8e8e' },
    '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
};

const modalSelectStyles = {
    ...modalTextFieldStyles,
    '& .MuiSvgIcon-root': { color: '#8e8e8e' },
    '& .MuiSelect-select': { color: '#e0e0e0' },
};

const providers = [
    { id: 'openai', name: 'OpenAI' },
    { id: 'gemini', name: 'Gemini' },
];

const ApiKeyInput = () => {
    const [apiKey, setApiKey] = useState('');
    const [isSaved, setIsSaved] = useState(false);

    const [openModal, setOpenModal] = useState(false);
    const [provider, setProvider] = useState('');
    const [keyName, setKeyName] = useState('');

    const navigate = useNavigate();

    const handleOpenModal = () => {
        if (apiKey.trim()) {
            setOpenModal(true);
        }
    };

    const handleModalClose = () => {
        setOpenModal(false);
    };

    async function handleFinalSave() {


        const data = {
            provider : provider,
            key_name : keyName,
            key : apiKey
        }
        try {
            const response = await axios.post('http://localhost:8000/store_model/store',data);

            if (response.status === 200) {
                setIsSaved(true);
            }

        }catch (error) {
            alert("Invalid Key")
            setIsSaved(false)

        }



        setOpenModal(false);
        setApiKey('');
        setProvider('');
        setKeyName('');



    }

    const handleBack = () => {
        navigate("/Choice");
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
                    overflowY: 'auto',
                    '& *': {
                        outline: 'none !important',
                    },
                    '& *:focus': {
                        outline: 'none !important',
                    },
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
                    type="password"
                    onChange={(e) => {
                        setApiKey(e.target.value);
                        setIsSaved(false);
                    }}
                    // --- THIS SECTION IS UPDATED ---
                    sx={{
                        mt: 1,
                        '& .MuiInputBase-input': {
                            color: '#e0e0e0',
                            letterSpacing: '1px'
                        },
                        // Style for the label
                        '& .MuiInputLabel-root': {
                            color: '#e0e0e0' // Unfocused label color
                        },
                        '& .MuiInputLabel-root.Mui-focused': {
                            color: '#1a73e8' // Focused label color
                        },
                        // Style for the border
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
                    }}
                />

                <Button
                    fullWidth
                    variant="contained"
                    onClick={handleOpenModal}
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
                    Save/Replace Key
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

            <Dialog
                open={openModal}
                onClose={handleModalClose}
                PaperProps={{
                    sx: {
                        bgcolor: '#1e2023',
                        color: '#e0e0e0',
                        borderRadius: 3,
                        border: '1px solid #3e4042',
                        boxShadow: '0 8px 30px rgba(0,0,0,0.7)',
                        width: '100%',
                        maxWidth: 400,
                    }
                }}
            >
                <DialogTitle sx={{ fontWeight: 600, letterSpacing: 0.5 }}>
                    Add Key Details
                </DialogTitle>
                <DialogContent>
                    <DialogContentText sx={{ color: '#8e8e8e', mb: 3 }}>
                        Please select the provider and give this key a nickname.
                    </DialogContentText>

                    <FormControl
                        fullWidth
                        variant="outlined"
                        sx={modalSelectStyles}
                    >
                        <InputLabel id="provider-select-label">Provider</InputLabel>
                        <Select
                            labelId="provider-select-label"
                            id="provider-select"
                            value={provider}
                            label="Provider"
                            onChange={(e) => setProvider(e.target.value)}
                            // Style the dropdown menu itself
                            MenuProps={{
                                PaperProps: {
                                    sx: {
                                        bgcolor: '#282a2e',
                                        color: '#e0e0e0',
                                        border: '1px solid #3e4042',
                                    },
                                },
                            }}
                        >
                            {providers.map((p) => (
                                <MenuItem
                                    key={p.id}
                                    value={p.id}
                                    sx={{
                                        '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.05)' },
                                        '&.Mui-selected': { bgcolor: 'rgba(26, 115, 232, 0.2)' },
                                        '&.Mui-selected:hover': { bgcolor: 'rgba(26, 115, 232, 0.3)' }
                                    }}
                                >
                                    {p.name}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>

                    <TextField
                        margin="dense"
                        id="keyName"
                        label="Key Name (Nickname)"
                        type="text"
                        fullWidth
                        variant="outlined"
                        value={keyName}
                        onChange={(e) => setKeyName(e.target.value)}
                        sx={{...modalTextFieldStyles, mt: 3}} // Added margin-top
                    />
                </DialogContent>
                <DialogActions sx={{ p: '16px 24px' }}>
                    <Button
                        onClick={handleModalClose}
                        sx={{ color: '#8e8e8e', '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.05)' } }}
                    >
                        Cancel
                    </Button>
                    <Button
                        onClick={handleFinalSave}
                        variant="contained"
                        // Disable button if any field is empty
                        disabled={!provider || !keyName.trim()}
                        sx={{
                            bgcolor: '#1a73e8',
                            '&:hover': { bgcolor: '#1565c0' },
                            '&.Mui-disabled': { bgcolor: '#282a2e', color: '#8e8e8e' }
                        }}
                    >
                        Save Details
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default ApiKeyInput;