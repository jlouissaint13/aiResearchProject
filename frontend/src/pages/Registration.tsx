import {useState} from 'react';
import { Box, Typography, TextField, Button, Link } from '@mui/material';
import PersonAddAltIcon from '@mui/icons-material/PersonAddAlt';
import {useNavigate} from "react-router-dom";

const Registration = () => {
    const navigate = useNavigate();

    const [firstName,setFirstName] = useState<string>('');
    const [email,setEmail] = useState<string>('');
    const [username,setUsername] = useState<string>('');
    const [password,setPassword] = useState<string>('');
    const [confirmPassword,setConfirmPassword] = useState<string>('');
    const [invalidEmail,setInvalidEmail] = useState<boolean>(false);
    const [passwordMatchError,setPasswordMatchError] = useState<boolean>(false);
    async function registration() {
        const data = {
            firstName: firstName.trim(),
            email: email.trim(),
            username: username.trim(),
            password: password.trim(),
            confirmPassword: confirmPassword.trim()
        }

        if (formIsEmpty(data)) {
            alert("Please fill out all required fields");
            return
        }


        if (!isEmailValid(email.trim())) {
            setInvalidEmail(true);
            return;
        }
        setInvalidEmail(false)




        if (!passwordMatches(password.trim(),confirmPassword.trim())) {
            setPasswordMatchError(true);
            return
        }
        setPasswordMatchError(false);



        try {
            const response = await fetch('http://localhost:8000/user/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
            if (response.status === 200) {
                clearFields()
                alert("Welcome "+ firstName.trim())
                navigate('/Login');

            }
            else if (response.status === 409) {
                alert("User already exists")
            }
        } catch (error) {
            console.log("error")
        }
    }

    function isEmailValid(email:string): boolean {
        if (email.includes('@') && email.includes('.com')) {
            setInvalidEmail(true)
            return true;
        }


        return false;
    }

    function formIsEmpty(data: Record<string, string>) : boolean {
        for(let i in data) {
            if (data[i].length == 0 || data[i] === "")
                return true;

        }
        return false;
    }
    function passwordMatches(password:string,confirmPassword:string): boolean {
        if (password !== confirmPassword)
            return false;

        return true;
    }

    const handleKeyPress = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter') {
            registration();
        }
    };


    function clearFields() {
        setUsername('');
        setPassword('');
        setEmail('');
        setFirstName('');
        setConfirmPassword('');
    }

    function haveAccount() {
        navigate("/login")
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
                    p: { xs: 3, md: 4 },
                    bgcolor: 'rgba(41, 43, 46, 0.8)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: 4,
                    boxShadow: '0px 8px 30px rgba(0, 0, 0, 0.6)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 3,
                    width: '100%',
                    maxWidth: 450,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
            >
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                    <PersonAddAltIcon sx={{ fontSize: 80, color: '#e0e0e0' }} />
                    <Typography variant="h5" component="h1" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>
                        Create Your Account
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#8e8e8e' }}>
                        Enter your details to get started
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="filled"
                    label="First Name"
                    onChange={event => setFirstName(event.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{
                        '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#3e4042' },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                    }}
                />

                <TextField
                    fullWidth
                    variant="filled"
                    label="Email"
                    type="email"
                    onChange={event => setEmail(event.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{
                        '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#3e4042' },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                    }}
                    error={invalidEmail}
                    helperText={invalidEmail ? "Enter a valid email.": ""}
                />

                <TextField
                    fullWidth
                    variant="filled"
                    label="Username"
                    onChange={event => setUsername(event.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{
                        '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#3e4042' },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                    }}
                />

                <TextField
                    fullWidth
                    variant="filled"
                    label="Password"
                    type="password"
                    onChange={event => setPassword(event.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{
                        '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#3e4042' },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                    }}

                />

                <TextField
                    fullWidth
                    variant="filled"
                    label="Confirm Password"
                    type="password"
                    onChange={event => setConfirmPassword(event.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{
                        '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#3e4042' },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                    }}
                    error={passwordMatchError}
                    helperText={passwordMatchError ? "Passwords do not match": ""}
                />

                <Button
                    fullWidth
                    variant="contained"
                    onClick={registration}
                    sx={{
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        textTransform: 'none',
                        '&:hover': { bgcolor: '#1565c0' },
                    }}
                >
                    Sign Up
                </Button>

                <Link onClick={haveAccount} href="#" variant="body2" sx={{ color: '#8e8e8e', '&:hover': { color: '#e0e0e0' } }}>
                    Already have an account? Login here
                </Link>
            </Box>
        </Box>
    );
};

export default Registration;