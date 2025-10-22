import {useState} from 'react';
import {Box, Button, Link, TextField, Typography} from '@mui/material';
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
       
        const firstNameCapitalized = capitalizeFirstName().trim();
        const data = {
            firstName: firstNameCapitalized,
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
                alert("Welcome "+ firstNameCapitalized)
                navigate('/Login');

            }
            else if (response.status === 409) {
                alert("User already exists")
            }
        } catch (error) {
            console.log("error")
        }
    }
    //replace with regex
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
    
    
    function capitalizeFirstName() : string {
        return firstName[0].toUpperCase() + firstName.slice(1);
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
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, mb: 1 }}>
                    <PersonAddAltIcon sx={{ fontSize: 60, color: '#1a73e8' }} />
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
                        Create Account
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#8e8e8e', mt: -1 }}>
                        Enter your details to get started
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="outlined"
                    label="First Name"
                    onChange={event => setFirstName(event.target.value)}
                    onKeyPress={handleKeyPress}
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
                    }}
                />

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Email"
                    type="email"
                    onChange={event => setEmail(event.target.value)}
                    onKeyPress={handleKeyPress}
                    error={invalidEmail}
                    helperText={invalidEmail ? "Enter a valid email.": ""}
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
                    label="Username"
                    onChange={event => setUsername(event.target.value)}
                    onKeyPress={handleKeyPress}
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
                    }}
                />

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Password"
                    type="password"
                    onChange={event => setPassword(event.target.value)}
                    onKeyPress={handleKeyPress}
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
                    }}
                />

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Confirm Password"
                    type="password"
                    onChange={event => setConfirmPassword(event.target.value)}
                    onKeyPress={handleKeyPress}
                    error={passwordMatchError}
                    helperText={passwordMatchError ? "Passwords do not match": ""}
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
                        '& .MMuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
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
                    onClick={registration}
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
                    Sign Up
                </Button>

                <Link onClick={haveAccount} href="#" variant="body2" sx={{ color: '#8e8e8e', '&:hover': { color: '#e0e0e0', textDecoration: 'none' } }}>
                    Already have an account? Login here
                </Link>
            </Box>
        </Box>
    );
};

export default Registration;