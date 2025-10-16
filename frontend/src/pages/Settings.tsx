import {useEffect, useState} from 'react';
import {
    Box, Typography, Button, TextField, Divider, List, ListItem,
    ListItemText, ListItemIcon, FormControl, FormLabel, RadioGroup, FormControlLabel, Radio,
    Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SettingsIcon from '@mui/icons-material/Settings';
import DeleteIcon from '@mui/icons-material/Delete';
import DnsIcon from '@mui/icons-material/Dns';
import { useNavigate } from "react-router-dom";

const initialModels = [
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash' },
    { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro' },
    { id: 'claude-3-opus', name: 'Claude 3 Opus' },
];
const initialDefaultModel = 'gemini-2.5-flash';
const initialTemperature = 0.7;
const initialTopP = 0.9;
const initialTopK = 40;

const inputStyle = {
    '& .MuiFilledInput-root': {
        borderRadius: 2,
        bgcolor: '#3e4042',
        '&:hover': { bgcolor: '#424549' },
        '&.Mui-focused': { bgcolor: '#3e4042' },
    },
    '& .MuiInputBase-input': { color: '#e0e0e0' },
    '& .MuiInputLabel-root': { color: '#8e8e8e' },
    '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
    '& .MuiFormHelperText-root.Mui-error': { color: '#ff7961' },
};

const buttonStyle = {
    p: 1.5,
    borderRadius: 2,
    bgcolor: '#1a73e8',
    color: '#fff',
    fontWeight: 'bold',
    textTransform: 'none',
    '&:hover': {
        bgcolor: '#1565c0',
        boxShadow: '0px 4px 15px rgba(26, 115, 232, 0.4)',
    },
};

const deleteButtonStyle = {
    p: 1.5,
    borderRadius: 2,
    bgcolor: '#d32f2f',
    color: '#fff',
    fontWeight: 'bold',
    textTransform: 'none',
    '&:hover': {
        bgcolor: '#b71c1c',
        boxShadow: '0px 4px 15px rgba(211, 47, 47, 0.4)',
    },
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

    const [temperature, setTemperature] = useState(initialTemperature);
    const [topP, setTopP] = useState(initialTopP);
    const [topK, setTopK] = useState(initialTopK);

    const [isPasswordDialogOpen, setIsPasswordDialogOpen] = useState(false);
    const [isDeleteConfirmOpen, setIsDeleteConfirmOpen] = useState(false);
    const [passwordConfirm, setPasswordConfirm] = useState('');

    const [usernameError, setUsernameError] = useState(false);
    const [emailError, setEmailError] = useState(false);
    const [invalidEmail, setInvalidEmail] = useState(false);

    const handleBack = () => {
        if (sessionStorage.getItem("lastPage") === "chatbot") {
            sessionStorage.clear();
            navigate("/Chatbot");
            return;
        }

        navigate("/Choice");
    };


    useEffect(() => {
        getUserInfo()
    },[]);

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

    const handleCloseDialog = () => {
        setIsPasswordDialogOpen(false);
        setPasswordConfirm('');
    };

    const handleCloseDeleteDialog = () => {
        setIsDeleteConfirmOpen(false);
        setPasswordConfirm('');
    };


    const handleSaveAdvanced = () => {
        console.log('Saving Advanced Settings:', { defaultModel, temperature, topP, topK });
        console.log('Advanced model settings saved!');
    };

    const handleDeleteModel = (modelIdToDelete) => {
        const updatedModels = models.filter(model => model.id !== modelIdToDelete);
        setModels(updatedModels);

        if (defaultModel === modelIdToDelete) {
            setDefaultModel(updatedModels.length > 0 ? updatedModels[0].id : '');
        }
    };

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
                justifyContent: 'center',
                position: 'absolute',
                inset: 0,
                background: 'linear-gradient(135deg, #1A2027 0%, #171A21 100%)',
                color: '#e0e0e0',
                fontFamily: 'Roboto, sans-serif',
                p: 3,
            }}
        >
            <Box sx={{ position: 'absolute', top: 24, left: 24, zIndex: 10 }}>
                <Button
                    onClick={handleBack}
                    variant="text"
                    startIcon={<ArrowBackIcon />}
                    sx={{ color: '#e0e0e0', textTransform: 'none', '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.08)' } }}
                >
                    Back
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
                    gap: 4,
                    width: '100%',
                    maxWidth: 650,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    maxHeight: '90vh',
                    overflowY: 'auto',
                }}
            >
                <Typography variant="h5" component="h1" sx={{ color: '#e0e0e0', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SettingsIcon /> Application Settings
                </Typography>

                <Box sx={{ width: '100%' }}>
                    <Typography variant="h6" sx={{ color: '#88aaff', mb: 2 }}>
                        Authentication
                    </Typography>
                    <TextField
                        fullWidth
                        variant="filled"
                        label="Username"
                        value={username}
                        onChange={(e) => {
                            setUsername(e.target.value);
                            setUsernameError(false);
                        }}
                        error={usernameError}
                        helperText={usernameError ? "Username cannot be empty." : ""}
                        sx={{ mb: 2, ...inputStyle }}
                    />
                    <TextField
                        fullWidth
                        variant="filled"
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
                        sx={{ mb: 2, ...inputStyle }}
                    />
                    <TextField fullWidth variant="filled" label="New Password (Leave Blank to Keep Current)" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} sx={{ mb: 3, ...inputStyle }} />
                    <Button fullWidth variant="contained" onClick={handleSaveGeneral} sx={buttonStyle}>
                        Save User Settings
                    </Button>
                </Box>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

                <Box sx={{ width: '100%' }}>
                    <Typography variant="h6" sx={{ color: '#88aaff', mb: 2 }}>
                        Advanced Model Parameters
                    </Typography>
                    <Box sx={{ p: 2, border: '1px solid #3e4042', borderRadius: 2, bgcolor: '#3e4042', display: 'flex', flexDirection: 'column', gap: 3 }}>
                        <FormControl component="fieldset">
                            <FormLabel component="legend" sx={{ color: '#8e8e8e', mb: 1, '&.Mui-focused': { color: '#8e8e8e' } }}>Default Model</FormLabel>
                            <RadioGroup value={defaultModel} onChange={(e) => setDefaultModel(e.target.value)}>
                                {models.map((model) => (
                                    <FormControlLabel key={model.id} value={model.id} control={<Radio sx={{ color: '#8e8e8e', '&.Mui-checked': { color: '#1a73e8' } }} />} label={model.name} />
                                ))}
                            </RadioGroup>
                        </FormControl>
                        <Typography variant="body2" sx={{ color: '#8e8e8e', mt: -1 }}>
                            Adjusting these parameters affects the creativity and randomness of the responses.
                        </Typography>
                        <TextField fullWidth variant="filled" label="Temperature (0.0 - 1.0)" type="number" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} inputProps={{ min: 0.0, max: 1.0, step: 0.1 }} sx={inputStyle} />
                        <TextField fullWidth variant="filled" label="Top P (Nucleus Sampling)" type="number" value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))} inputProps={{ min: 0.0, max: 1.0, step: 0.05 }} sx={inputStyle} />
                        <TextField fullWidth variant="filled" label="Top K (Token Selection)" type="number" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} inputProps={{ min: 1, step: 1 }} sx={inputStyle} />
                        <Button fullWidth variant="contained" onClick={handleSaveAdvanced} sx={buttonStyle}>
                            Save Model Configuration
                        </Button>
                    </Box>
                </Box>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

                <Box sx={{ width: '100%' }}>
                    <Typography variant="h6" sx={{ color: '#88aaff', mb: 2 }}>
                        Model Manager
                    </Typography>
                    <Box sx={{ p: 2, border: '1px solid #3e4042', borderRadius: 2, bgcolor: '#3e4042' }}>
                        <Typography variant="body2" sx={{ color: '#8e8e8e', mb: 2 }}>
                            View and remove available models from the application.
                        </Typography>
                        <List>
                            {models.length > 0 ? (
                                models.map((model) => (
                                    <ListItem
                                        key={model.id}
                                        sx={{ bgcolor: '#424549', borderRadius: 2, mb: 1, '&:hover': { bgcolor: '#4f5257' } }}
                                        secondaryAction={
                                            <Button onClick={() => handleDeleteModel(model.id)} sx={{ color: '#ff7961' }} startIcon={<DeleteIcon />}>
                                                Delete
                                            </Button>
                                        }
                                    >
                                        <ListItemIcon sx={{ color: '#e0e0e0', minWidth: '40px' }}><DnsIcon /></ListItemIcon>
                                        <ListItemText primary={model.name} primaryTypographyProps={{ color: '#e0e0e0', fontWeight: '500' }} />
                                    </ListItem>
                                ))
                            ) : (
                                <Typography sx={{ textAlign: 'center', color: '#8e8e8e', fontStyle: 'italic', mt: 2 }}>
                                    No models available.
                                </Typography>
                            )}
                        </List>
                    </Box>
                </Box>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

                <Box sx={{ width: '100%' }}>
                    <Typography variant="h6" sx={{ color: '#ff7961', mb: 2 }}>
                        Account Deletion
                    </Typography>
                    <Box sx={{ p: 2, border: '1px solid #3e4042', borderRadius: 2, bgcolor: '#3e4042' }}>
                        <Typography variant="body2" sx={{ color: '#8e8e8e', mb: 2 }}>
                            Permanently delete your account and all associated data, including chat history and user information. This action is irreversible.
                        </Typography>
                        <Button
                            fullWidth
                            variant="contained"
                            onClick={handleDeleteAccountClick}
                            sx={deleteButtonStyle}
                        >
                            Delete Account
                        </Button>
                    </Box>
                </Box>
            </Box>

            <Dialog open={isPasswordDialogOpen} onClose={handleCloseDialog} PaperProps={{ sx: { bgcolor: '#3e4042', color: '#e0e0e0', borderRadius: 2, border: '1px solid rgba(255, 255, 255, 0.1)', p:1 } }}>
                <DialogTitle sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>Confirm Changes</DialogTitle>
                <DialogContent>
                    <Typography sx={{ mb: 2 }}>
                        Please enter your current password to confirm.
                    </Typography>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Current Password"
                        type="password"
                        fullWidth
                        variant="filled"
                        value={passwordConfirm}
                        onChange={(e) => setPasswordConfirm(e.target.value)}
                        sx={inputStyle}
                        InputLabelProps={{ shrink: true }}
                    />
                </DialogContent>
                <DialogActions sx={{ p: 2 }}>
                    <Button onClick={handleCloseDialog} sx={{ color: '#8e8e8e', textTransform: 'none' }}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleConfirmPasswordAndSave}
                        disabled={!passwordConfirm}
                        sx={{ ...buttonStyle, minWidth: 100 }}
                    >
                        Confirm & Save
                    </Button>
                </DialogActions>
            </Dialog>

            <Dialog
                open={isDeleteConfirmOpen}
                onClose={handleCloseDeleteDialog}
                PaperProps={{ sx: { bgcolor: '#3e4042', color: '#e0e0e0', borderRadius: 2, border: '1px solid rgba(255, 255, 255, 0.1)', p:1 } }}
            >
                <DialogTitle sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>Confirm Deletion</DialogTitle>
                <DialogContent>
                    <Typography sx={{ mb: 2 }}>
                        Please enter your current password to confirm permanent account deletion.
                    </Typography>
                    <TextField
                        autoFocus
                        margin="dense"
                        label="Current Password"
                        type="password"
                        fullWidth
                        variant="filled"
                        value={passwordConfirm}
                        onChange={(e) => setPasswordConfirm(e.target.value)}
                        sx={inputStyle}
                        InputLabelProps={{ shrink: true }}
                    />
                </DialogContent>
                <DialogActions sx={{ p: 2 }}>
                    <Button onClick={handleCloseDeleteDialog} sx={{ color: '#8e8e8e', textTransform: 'none' }}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleConfirmPasswordAndDelete}
                        disabled={!passwordConfirm}
                        sx={deleteButtonStyle}
                    >
                        Confirm & Delete
                    </Button>
                </DialogActions>
            </Dialog>

        </Box>
    );
};

export default Settings;