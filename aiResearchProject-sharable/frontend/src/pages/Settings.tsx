import {useEffect, useState} from 'react';
import {
    Box, Typography, Button, TextField, Divider, FormControl, FormLabel, RadioGroup, FormControlLabel, Radio,
    Dialog, DialogTitle, DialogContent, DialogActions, Tooltip,
    Select, MenuItem, InputLabel, LinearProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SettingsIcon from '@mui/icons-material/Settings';
import { useNavigate } from "react-router-dom";
import axios from "axios";

const initialModels = [
    { id: '', name: '', provider: '' },
];


const initialDefaultModel = 'gemini-2.5-flash';
const initialPromptType = 'deep-research';

const textFieldStyle = {
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
    '& .MuiFormHelperText-root': {
        color: '#8e8e8e',
        '&.Mui-error': {
            color: '#f44336',
        },
    },
};

const primaryButtonStyle = {
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
    '&:disabled': {
        bgcolor: '#3e4042',
        color: '#8e8e8e',
        transform: 'none',
        boxShadow: 'none',
        cursor: 'not-allowed',
    }
};

const deleteButtonStyle = {
    ...primaryButtonStyle,
    bgcolor: '#f44336',
    '&:hover': {
        bgcolor: '#d32f2f',
        transform: 'translateY(-1px)',
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)',
    },
    '&:disabled': {
        bgcolor: '#3e4042',
        color: '#8e8e8e',
        transform: 'none',
        boxShadow: 'none',
        cursor: 'not-allowed',
    }
};

const promptDescriptions = {
    'deep-research': 'Provides a thorough, structured, and in-depth answer. It explains connections and sticks strictly to the provided facts.',
    'creative': 'Generates a thoughtful and imaginative response. It connects ideas, draws new insights, and explores different angles based on the text.',
    'short-and-sweet': 'Delivers a concise, clear, and direct answer. It\'s accurate and gets right to the point, avoiding any filler.'
};

const Settings = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [newPassword, setNewPassword] = useState('');

    const [initialUsername,setInitialUsername] = useState('');
    // @ts-ignore
    const [initialEmail,setInitialEmail] = useState('');

    const navigate = useNavigate();

    const [models, setModels] = useState(initialModels);
    const [defaultModel, setDefaultModel] = useState(initialDefaultModel);

    const [promptType, setPromptType] = useState(initialPromptType);

    const [isPasswordDialogOpen, setIsPasswordDialogOpen] = useState(false);
    const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
    const [passwordConfirm, setPasswordConfirm] = useState('');

    const [usernameError, setUsernameError] = useState(false);
    const [emailError, setEmailError] = useState(false);
    const [invalidEmail, setInvalidEmail] = useState(false);

    const [isLoading, setIsLoading] = useState(true);

    const handleBack = () => {
        if (sessionStorage.getItem("lastPage") === "chatbot") {
            sessionStorage.clear();
            navigate("/Chatbot");
            return;
        }
        else if (sessionStorage.getItem("lastPage") === "chatbotTemporary") {
            sessionStorage.clear()
            navigate("/chatbotTemp")
            return;
        }

        navigate("/Choice");
    };




    async function getUserInfo() : Promise<void> {
        const data = {
            user_id: localStorage.getItem("userID")
        }

        try{
            const response = await fetch('http://localhost:8000/user_settings/retrieve_user_info', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)

            });
            if (response.status === 200) {
                const userInfo = await response.json();

                setUsername(userInfo.username);
                setEmail(userInfo.email);

                setInitialEmail(userInfo.email);
                setInitialUsername(userInfo.username);
            }

        }catch (error) {
            alert(error)
        }
    }

    const isEmailValid = (email:string): boolean => {
        return email.includes('@') && email.includes('.com');
    }

    const handleSaveGeneral = () => {
        let hasError = false;

        if (!username.trim()) {
            setUsernameError(true);
            hasError = true;
        } else {
            setUsernameError(false);
        }

        if (!email.trim()) {
            setEmailError(true);
            setInvalidEmail(false);
            hasError = true;
        } else {
            setEmailError(false);

            if (!isEmailValid(email)) {
                setInvalidEmail(true);
                hasError = true;
            } else {
                setInvalidEmail(false);
            }
        }

        if (hasError) {
            return;
        }

        setIsPasswordDialogOpen(true);
    };

    async function updateData(data: any) : Promise<void>{
        try {
            const response = await fetch('http://localhost:8000/user_settings/update_user_info', {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (response.status === 409) {
                alert("The new username or email address is already in use by another account.");
                setPasswordConfirm('');
                setNewPassword('');
                return;
            }

            if (response.status === 200) {
                alert("Your profile changes have been saved.");

                setIsPasswordDialogOpen(false);
                setPasswordConfirm('');
                setNewPassword('');

                setInitialUsername(data.username);
                setInitialEmail(data.email);
            }
        }catch (error) {
            alert(error);
        }
    }

    async function handleConfirmPasswordAndSave() : Promise<void> {
        const authData = {
            username: initialUsername.trim(),
            password: passwordConfirm.trim()
        };

        try {
            const authResponse = await fetch('http://localhost:8000/login/auth', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(authData),
            });

            if (authResponse.status === 401) {
                alert("Invalid password. Please enter your current password to confirm changes.");
                setPasswordConfirm('');
                return;
            }

            if (authResponse.status === 200) {

                const newData = {
                    user_id: localStorage.getItem("userID"),
                    username: username.trim(),
                    email : email.trim(),
                    password : newPassword.trim()
                }

                await updateData(newData)
            }

        } catch (error) {
            alert(error);
        }
    }

    async function handleConfirmPasswordAndDelete() : Promise<void> {
        const authData = {
            username: initialUsername.trim(),
            password: passwordConfirm.trim()
        };

        try {
            const authResponse = await fetch('http://localhost:8000/login/auth', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(authData),
            });

            if (authResponse.status === 401) {
                alert("Invalid password. Please enter your current password to confirm deletion.");
                setPasswordConfirm('');
                return;
            }

            if (authResponse.status === 200) {

                const deleteData = {
                    user_id: localStorage.getItem("userID")
                };

                const deleteResponse = await fetch('http://localhost:8000/user_settings/delete/user_account', {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(deleteData),
                });

                if (deleteResponse.status === 200) {
                    alert("Your account has been permanently deleted.");
                    localStorage.clear();
                    sessionStorage.clear();
                    navigate("/Login");
                }

                setIsDeleteConfirmOpen(false);
                setPasswordConfirm('');
            }

        } catch (error) {
            alert(error);
        }
    }


    // @ts-ignore
    async function retrieveUserSettings(setDefaultModel, setPromptType) {
        const userId : string | null = localStorage.getItem("userID");


        try {
            const response = await axios.post('http://localhost:8000/model_settings/retrieve_settings', {
                user_id: userId
            });

            if (response.status === 200 && response.data) {
                const modelSettings = response.data;
                if (modelSettings.activeModel) {
                    setDefaultModel(modelSettings.activeModel);
                }
                if (modelSettings.promptType) {
                    setPromptType(modelSettings.promptType);
                }
            } else if (response.status === 404) {
                return
            }
        } catch (error : any) {
            if (error.response && error.response.status === 404) {

            } else {
                console.error("An error occurred while retrieving user settings:", error);
            }
        }
    }

    useEffect(() =>  {
        const loadData = async () => {
            setIsLoading(true);
            try {
                await Promise.all([
                    retrieveModels(),
                    retrieveUserSettings(setDefaultModel, setPromptType),
                    getUserInfo()
                ]);
            } catch (error) {
                console.error("Failed to load initial data:", error);
            } finally {
                setIsLoading(false);
            }
        };

        loadData();
    }, []);





    const handleCloseDialog = () => {
        setIsPasswordDialogOpen(false);
        setPasswordConfirm('');
    };

    const handleCloseDeleteDialog = () => {
        setIsDeleteConfirmOpen(false);
        setPasswordConfirm('');
    };


    async function handleSaveConfiguration(): Promise<void> {
        const selectedModelObject = models.find(model => model.id === defaultModel)
        const provider = selectedModelObject?.provider;
        const data = {
            user_id : localStorage.getItem("userID"),
            prompt_type : promptType,
            active_model: defaultModel,
            provider : provider

        }
        try {
            const response = await axios.post("http://localhost:8000/model_settings/save_model_settings",data)

            if (response.status === 200) {
                alert("Model Settings Saved")
            }

        }catch (error) {
            alert(error)
        }

    }



    async function retrieveModels() : Promise<void> {
        try {
            const response = await axios.get('http://localhost:8000/model_settings/retrieve_models');
            const fetchedModels = response.data;
            if (response.status === 200) {
                setModels(fetchedModels);
            }

        } catch (error) {
            alert(error)
            setModels([]);
        }
    }


    const handleDeleteAccountClick = () => {


        if (!window.confirm("Are you absolutely sure you want to permanently delete your account? This action cannot be undone and all associated data will be lost.")) {
            return;
        }

        setIsDeleteConfirmOpen(true);
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'flex-start',
                position: 'absolute',
                inset: 0,
                background: 'linear-gradient(135deg, #1A2027 0%, #171A21 100%)',
                color: '#e0e0e0',
                fontFamily: 'Roboto, sans-serif',
                p: 3,
                overflowY: 'auto',
            }}
        >
            <Box sx={{
                position: 'absolute',
                top: { xs: 16, md: 20 },
                left: { xs: 16, md: 20 },
                zIndex: 10,
            }}>
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
                    Back
                </Button>
            </Box>

            <Box
                sx={{
                    position: 'relative',
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
                    maxWidth: 650,
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    mt: { xs: 6, md: 4 },
                    mb: 4,
                }}
            >
                {isLoading && (
                    <LinearProgress
                        sx={{
                            width: '100%',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            bgcolor: 'transparent',
                            '& .MuiLinearProgress-bar': {
                                bgcolor: '#1a73e8'
                            }
                        }}
                    />
                )}
                <Typography
                    variant="h5"
                    component="h1"
                    sx={{
                        color: '#e0e0e0',
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1.5,
                        letterSpacing: 0.5,
                        textTransform: 'uppercase',
                    }}
                >
                    <SettingsIcon sx={{ color: '#1a73e8', fontSize: '28px' }} />
                    Application Settings
                </Typography>

                <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Typography variant="h6" sx={{ color: '#1a73e8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, mb: -1 }}>
                        Authentication
                    </Typography>
                    <TextField
                        fullWidth
                        variant="outlined"
                        label="Username"
                        value={username}
                        onChange={(e) => {
                            setUsername(e.target.value);
                            setUsernameError(false);
                        }}
                        error={usernameError}
                        helperText={usernameError ? "Username cannot be empty." : ""}
                        sx={textFieldStyle}
                        disabled={isLoading}
                    />
                    <TextField
                        fullWidth
                        variant="outlined"
                        label="Email Address"
                        type="email"
                        value={email}
                        onChange={(e) => {
                            setEmail(e.target.value);
                            if (emailError || invalidEmail) {
                                setEmailError(false);
                                setInvalidEmail(false);
                            }
                        }}
                        error={emailError || invalidEmail}
                        helperText={
                            emailError ? "Email Address cannot be empty." :
                                invalidEmail ? "Please enter a valid email" :
                                    ""
                        }
                        sx={textFieldStyle}
                        disabled={isLoading}
                    />
                    <TextField fullWidth variant="outlined" label="New Password (Leave Blank to Keep Current)" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} sx={textFieldStyle} disabled={isLoading} />
                    <Button fullWidth variant="contained" onClick={handleSaveGeneral} sx={primaryButtonStyle} disabled={isLoading}>
                        Save User Settings
                    </Button>
                </Box>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.08)' }} />


                <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Typography variant="h6" sx={{ color: '#1a73e8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, mb: -1 }}>
                        Model & Prompt Settings
                    </Typography>
                    <Box sx={{ p: 3, border: '1px solid #3e4042', borderRadius: 1, bgcolor: '#282a2e', display: 'flex', flexDirection: 'column', gap: 3 }}>


                        <FormControl fullWidth variant="outlined" disabled={isLoading}>
                            <InputLabel
                                id="default-model-select-label"
                                sx={{
                                    color: '#8e8e8e',
                                    '&.Mui-focused': { color: '#1a73e8' },
                                }}
                            >
                                Current Model
                            </InputLabel>
                            <Select
                                labelId="default-model-select-label"
                                id="default-model-select"
                                value={defaultModel}
                                onChange={(e) => setDefaultModel(e.target.value)}
                                label="Default Model"
                                sx={{
                                    color: '#e0e0e0',
                                    bgcolor: '#282a2e',
                                    borderRadius: 1,
                                    '& .MuiOutlinedInput-notchedOutline': {
                                        borderColor: '#3e4042',
                                        transition: 'border-color 0.3s',
                                    },
                                    '&:hover .MuiOutlinedInput-notchedOutline': {
                                        borderColor: '#5e6062',
                                    },
                                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                                        borderColor: '#1a73e8',
                                        borderWidth: '2px',
                                    },
                                    '& .MuiSvgIcon-root': {
                                        color: '#8e8e8e',
                                    },
                                    '&.Mui-disabled': {
                                        color: '#8e8e8e',
                                        bgcolor: '#282a2e',
                                    },
                                    '&.Mui-disabled .MuiOutlinedInput-notchedOutline': {
                                        borderColor: '#3e4042',
                                    },
                                    '&.Mui-disabled .MuiSvgIcon-root': {
                                        color: '#3e4042',
                                    },
                                }}
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
                                {models.map((model) => (
                                    <MenuItem
                                        key={model.id}
                                        value={model.id}
                                        sx={{
                                            '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.08)' },
                                            '&.Mui-selected': { bgcolor: 'rgba(26, 115, 232, 0.2)' },
                                            '&.Mui-selected:hover': { bgcolor: 'rgba(26, 115, 232, 0.3)' },
                                        }}
                                    >
                                        {model.name.toUpperCase()}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <Divider sx={{ my: 1, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />


                        <FormControl component="fieldset" disabled={isLoading}>
                            <FormLabel component="legend" sx={{ color: '#8e8e8e', mb: 1, '&.Mui-focused': { color: '#8e8e8e' } }}>Prompt Type</FormLabel>
                            <RadioGroup value={promptType} onChange={(e) => setPromptType(e.target.value)}>
                                <Tooltip title={promptDescriptions['deep-research']} placement="right">
                                    <FormControlLabel
                                        value="deep-research"
                                        control={<Radio sx={{ color: '#8e8e8e', '&.Mui-checked': { color: '#1a73e8' } }} />}
                                        label="Deep Research"
                                    />
                                </Tooltip>
                                <Tooltip title={promptDescriptions['creative']} placement="right">
                                    <FormControlLabel
                                        value="creative"
                                        control={<Radio sx={{ color: '#8e8e8e', '&.Mui-checked': { color: '#1a73e8' } }} />}
                                        label="Creative"
                                    />
                                </Tooltip>
                                <Tooltip title={promptDescriptions['short-and-sweet']} placement="right">
                                    <FormControlLabel
                                        value="short-and-sweet"
                                        control={<Radio sx={{ color: '#8e8e8e', '&.Mui-checked': { color: '#1a73e8' } }} />}
                                        label="Short and Sweet"
                                    />
                                </Tooltip>
                            </RadioGroup>
                        </FormControl>

                        <Button fullWidth variant="contained" onClick={handleSaveConfiguration} sx={primaryButtonStyle} disabled={isLoading}>
                            Save Configuration
                        </Button>
                    </Box>
                </Box>



                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

                <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Typography variant="h6" sx={{ color: '#f44336', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, mb: -1 }}>
                        Account Deletion
                    </Typography>
                    <Box sx={{ p: 3, border: '1px solid #3e4042', borderRadius: 1, bgcolor: '#282a2e' }}>
                        <Typography variant="body2" sx={{ color: '#8e8e8e', mb: 2 }}>
                            Permanently delete your account and all associated data, including chat history and user information. This action is irreversible.
                        </Typography>
                        <Button
                            fullWidth
                            variant="contained"
                            onClick={handleDeleteAccountClick}
                            sx={deleteButtonStyle}
                            disabled={isLoading}
                        >
                            Delete Account
                        </Button>
                    </Box>
                </Box>
            </Box>

            <Dialog
                open={isPasswordDialogOpen}
                onClose={handleCloseDialog}
                PaperProps={{
                    sx: {
                        bgcolor: 'rgba(30, 32, 35, 0.98)',
                        color: '#e0e0e0',
                        borderRadius: 3,
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        p: 3,
                        boxShadow: '0 8px 30px rgba(0,0,0,0.7)',
                    }
                }}
            >
                <DialogTitle sx={{ color: '#e0e0e0', fontWeight: 600, p: 0, mb: 2 }}>Confirm Changes</DialogTitle>
                <DialogContent sx={{ p: 0 }}>
                    <Typography sx={{ mb: 3, color: '#8e8e8e' }}>
                        Please enter your current password to confirm.
                    </Typography>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Current Password"
                        type="password"
                        fullWidth
                        variant="outlined"
                        value={passwordConfirm}
                        onChange={(e) => setPasswordConfirm(e.target.value)}
                        sx={textFieldStyle}
                        InputLabelProps={{ shrink: true }}
                    />
                </DialogContent>
                <DialogActions sx={{ p: 0, pt: 3, gap: 1 }}>
                    <Button
                        onClick={handleCloseDialog}
                        sx={{
                            color: '#8e8e8e',
                            textTransform: 'none',
                            fontWeight: 500,
                            borderRadius: 1,
                            p: '8px 16px',
                            '&:hover': { bgcolor: '#282a2e' }
                        }}
                    >
                        Cancel
                    </Button>
                    <Button
                        onClick={handleConfirmPasswordAndSave}
                        disabled={!passwordConfirm}
                        sx={{ ...primaryButtonStyle, p: '8px 16px' }}
                    >
                        Confirm & Save
                    </Button>
                </DialogActions>
            </Dialog>

            <Dialog
                open={isDeleteConfirmOpen}
                onClose={handleCloseDeleteDialog}
                PaperProps={{
                    sx: {
                        bgcolor: 'rgba(30, 32, 35, 0.98)',
                        color: '#e0e0e0',
                        borderRadius: 3,
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        p: 3,
                        boxShadow: '0 8px 30px rgba(0,0,0,0.7)',
                    }
                }}
            >
                <DialogTitle sx={{ color: '#e0e0e0', fontWeight: 'bold', p: 0, mb: 2 }}>Confirm Deletion</DialogTitle>
                <DialogContent sx={{ p: 0 }}>
                    <Typography sx={{ mb: 3, color: '#8e8e8e' }}>
                        Please enter your current password to confirm permanent account deletion.
                    </Typography>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Current Password"
                        type="password"
                        fullWidth
                        variant="outlined"
                        value={passwordConfirm}
                        onChange={(e) => setPasswordConfirm(e.target.value)}
                        sx={textFieldStyle}
                        InputLabelProps={{ shrink: true }}
                    />
                </DialogContent>
                <DialogActions sx={{ p: 0, pt: 3, gap: 1 }}>
                    <Button
                        onClick={handleCloseDeleteDialog}
                        sx={{
                            color: '#8e8e8e',
                            textTransform: 'none',
                            fontWeight: 500,
                            borderRadius: 1,
                            p: '8px 16px',
                            '&:hover': { bgcolor: '#282a2e' }
                        }}
                    >
                        Cancel
                    </Button>
                    <Button
                        onClick={handleConfirmPasswordAndDelete}
                        disabled={!passwordConfirm}
                        sx={{ ...deleteButtonStyle, p: '8px 16px' }}
                    >
                        Confirm & Delete
                    </Button>
                </DialogActions>
            </Dialog>

        </Box>
    );
};

export default Settings;