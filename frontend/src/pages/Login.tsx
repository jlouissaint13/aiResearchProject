import { Box, Typography, TextField, Button, Link } from '@mui/material';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import {useNavigate} from "react-router-dom";
import {useState} from "react";

const Login = () => {
    const navigate = useNavigate()

    const [username,setUsername] = useState<string>('');
    const [password,setPassword] = useState<string>('');
    const [accountNotExists,setAccountNotExists] = useState<boolean>(false);
    const [invalidPassword,setInvalidPassword] = useState<boolean>(false);
    async function loginAccount() {


        const data = {
            //will take username or email so username will act as both
            username: username.trim(),
            password: password.trim()
        };
        if (formIsEmpty(data)) {
            alert("Please fill out all required fields");
            return;
        }

        try {
            const response = await fetch('http://localhost:8000/login/auth', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            if (response.status === 404) {
                setAccountNotExists(true)
                return;

            }
            setAccountNotExists(false);
            if (response.status === 401) {
                setInvalidPassword(true);
                return;
            }
            setInvalidPassword(false);


            if (response.status === 200) {
                clearFields()
                const res = await getUsernameID(username)
                localStorage.setItem("userID",res.userID)
                localStorage.setItem("loggedIn","true")
                localStorage.setItem("firstName",res.firstName)
                navigate('/Choice');

            }

        } catch (error) {
            console.log("error")
            //alert("error")
        }

    }

    //first index will be id second will be username
    async function getUsernameID(username:string) : Promise<any> {
        const info = {
            username: username
        }

        try {
            const response = await fetch('http://localhost:8000/login/user_logged', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(info),
            });

            if (response.status === 200) {

                const jsonResponse = await response.json();

                return jsonResponse
            }

        } catch (error) {
            console.log("error")
            //alert("error")
        }
        return "invalid"
    }





    function clearFields() {
        setUsername('');
        setPassword('')
    }


    function formIsEmpty(data: Record<string, string>) : boolean {
        for(let i in data) {
            if (data[i].length == 0 || data[i] === "")
                return true;

        }
        return false;
    }


    const handleKeyPress = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter') {
            loginAccount();
        }
    };


    function register() {
        navigate("/Registration")
    }

    function continueAsGuest() {
        localStorage.setItem("logged_in","false")
        navigate("/Choice")
    }







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
                    maxWidth: 400,
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                }}
            >
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                    <AccountCircleIcon sx={{ fontSize: 60, color: '#1a73e8' }} />
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
                        Welcome Back
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#8e8e8e', mt: -1 }}>
                        Log in to access your account
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Email or Username"
                    type="text"
                    onChange={event => setUsername(event.target.value)}
                    onKeyPress={handleKeyPress}
                    error={accountNotExists}
                    helperText={accountNotExists ? "User not found" : ""}
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
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                        '& .MuiFormHelperText-root': {
                            color: '#8e8e8e',
                            '&.Mui-error': {
                                color: '#f44336',
                            },
                        },
                    }}
                />

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Password"
                    type="password"
                    onChange={event => setPassword(event.target.value)}
                    onKeyPress={handleKeyPress}
                    error={invalidPassword}
                    helperText={invalidPassword ? "Invalid email/username or password" : ""}
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
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                        '& .MuiFormHelperText-root': {
                            color: '#8e8e8e',
                            '&.Mui-error': {
                                color: '#f44336',
                            },
                        },
                    }}
                />

                <Button
                    fullWidth
                    variant="contained"
                    onClick={loginAccount}
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
                    }}

                >
                    Login
                </Button>

                <Button
                    fullWidth
                    variant="contained"
                    onClick={continueAsGuest}
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
                    }}
                >

                    Continue as Guest
                </Button>

                <Box sx={{ display: 'flex', gap: 2, mt: 1, alignItems: 'center', justifyContent: 'center' }}>
                    <Link href="#" variant="body2" sx={{ color: '#8e8e8e', '&:hover': { color: '#e0e0e0', textDecoration: 'none' } }}>
                        Forgot password?
                    </Link>
                    <Link onClick={register} href="#" variant="body2" sx={{ color: '#8e8e8e', '&:hover': { color: '#e0e0e0', textDecoration: 'none' } }}>
                        Don't have an account? Sign up
                    </Link>
                </Box>
            </Box>
        </Box>
    );
};

export default Login;