import {useEffect, useState} from 'react';
import {
    Box, Typography, Button, TextField, Divider, List, ListItem,
    ListItemText, ListItemIcon, FormControl, FormLabel, RadioGroup, FormControlLabel, Radio,
    Dialog, DialogTitle, DialogContent, DialogActions, IconButton
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
};

const deleteButtonStyle = {
    ...primaryButtonStyle,
    bgcolor: '#f44336',
    '&:hover': {
        bgcolor: '#d32f2f',
        transform: 'translateY(-1px)',
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)',
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
    //add regex later
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
                    />
                    <TextField fullWidth variant="outlined" label="New Password (Leave Blank to Keep Current)" type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} sx={textFieldStyle} />
                    <Button fullWidth variant="contained" onClick={handleSaveGeneral} sx={primaryButtonStyle}>
                        Save User Settings
                    </Button>
                </Box>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

                <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Typography variant="h6" sx={{ color: '#1a73e8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, mb: -1 }}>
                        Advanced Model Parameters
                    </Typography>
                    <Box sx={{ p: 3, border: '1px solid #3e4042', borderRadius: 1, bgcolor: '#282a2e', display: 'flex', flexDirection: 'column', gap: 3 }}>
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
                        <TextField fullWidth variant="outlined" label="Temperature (0.0 - 1.0)" type="number" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} inputProps={{ min: 0.0, max: 1.0, step: 0.1 }} sx={textFieldStyle} />
                        <TextField fullWidth variant="outlined" label="Top P (Nucleus Sampling)" type="number" value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))} inputProps={{ min: 0.0, max: 1.0, step: 0.05 }} sx={textFieldStyle} />
                        <TextField fullWidth variant="outlined" label="Top K (Token Selection)" type="number" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} inputProps={{ min: 1, step: 1 }} sx={textFieldStyle} />
                        <Button fullWidth variant="contained" onClick={handleSaveAdvanced} sx={primaryButtonStyle}>
                            Save Model Configuration
                        </Button>
                    </Box>
                </Box>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

                <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Typography variant="h6" sx={{ color: '#1a73e8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, mb: -1 }}>
                        Model Manager
                    </Typography>
                    <Box sx={{ p: 3, border: '1px solid #3e4042', borderRadius: 1, bgcolor: '#282a2e' }}>
                        <Typography variant="body2" sx={{ color: '#8e8e8e', mb: 2 }}>
                            View and remove available models from the application.
                        </Typography>
                        <List>
                            {models.length > 0 ? (
                                models.map((model) => (
                                    <ListItem
                                        key={model.id}
                                        sx={{
                                            bgcolor: '#282a2e',
                                            borderRadius: 1,
                                            mb: 1.5,
                                            transition: 'background-color 0.3s',
                                            border: '1px solid #3e4042',
                                            '&:hover': {
                                                bgcolor: '#424549',
                                                borderColor: '#1a73e8'
                                            }
                                        }}
                                        secondaryAction={
                                            <IconButton
                                                onClick={() => handleDeleteModel(model.id)}
                                                sx={{
                                                    color: '#f44336',
                                                    '&:hover': { bgcolor: 'rgba(244, 67, 54, 0.1)', color: '#ff7961' },
                                                    '&:active': { bgcolor: 'transparent' },
                                                    outline: 'none',
                                                    '&:focus, &.Mui-focusVisible': {
                                                        bgcolor: 'transparent',
                                                        boxShadow: 'none',
                                                        outline: 'none'
                                                    }
                                                }}
                                                disableRipple
                                                disableFocusRipple
                                                disableTouchRipple
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        }
                                    >
                                        <ListItemIcon sx={{ color: '#1a73e8', minWidth: '40px' }}><DnsIcon /></ListItemIcon>
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