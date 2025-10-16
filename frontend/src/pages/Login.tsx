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
                    maxWidth: 450,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
            >
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                    <AccountCircleIcon sx={{ fontSize: 80, color: '#e0e0e0' }} />
                    <Typography variant="h5" component="h1" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>
                        Welcome Back
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#8e8e8e' }}>
                        Log in to access your account
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="filled"
                    label="Email or Username"
                    type="text"
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 2,
                            bgcolor: '#3e4042',
                            '& fieldset': { borderColor: 'transparent' },
                            '&:hover fieldset': { borderColor: '#5e5e5e' },
                            '&.Mui-focused fieldset': { borderColor: '#1a73e8' },
                        },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                    }}
                    onChange={event => setUsername(event.target.value)}
                    onKeyPress={handleKeyPress}
                    error={accountNotExists}
                    helperText={accountNotExists ? "User not found" : ""}
                />

                <TextField
                    fullWidth
                    variant="filled"
                    label="Password"
                    type="password"
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 2,
                            bgcolor: '#3e4042',
                            '& fieldset': { borderColor: 'transparent' },
                            '&:hover fieldset': { borderColor: '#5e5e5e' },
                            '&.Mui-focused fieldset': { borderColor: '#1a73e8' },
                        },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                    }}
                    onChange={event => setPassword(event.target.value)}
                    onKeyPress={handleKeyPress}
                    error={invalidPassword}
                    helperText={invalidPassword ? "Invalid email/username or password" : ""}
                />

                <Button
                    fullWidth
                    variant="contained"
                    onClick={loginAccount}
                    sx={{
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        textTransform: 'none',
                        '&:hover': { bgcolor: '#1565c0' },
                    }}

                >
                    Login
                </Button>

                <Button
                    fullWidth
                    variant="contained"
                    onClick={continueAsGuest}
                    sx={{
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        textTransform: 'none',
                        '&:hover': { bgcolor: '#1565c0' },
                    }}
                >

                    Continue as Guest
                </Button>

                <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: { xs: 1, sm: 2 }, mt: 1, alignItems: 'center' }}>
                    <Link href="#" variant="body2" sx={{ color: '#8e8e8e', '&:hover': { color: '#e0e0e0' } }}>
                        Forgot password?
                    </Link>
                    <Link onClick={register} href="#" variant="body2" sx={{ color: '#8e8e8e', '&:hover': { color: '#e0e0e0' } }}>
                        Don't have an account? Sign up
                    </Link>
                </Box>
            </Box>
        </Box>
    );
};

export default Login;