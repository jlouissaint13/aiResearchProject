import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Box, Typography, TextField, Button } from "@mui/material";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";
import ArrowBackIcon from "@mui/icons-material/ArrowBack"; // Added back

export default function VerifyCode() {
    const [code, setCode] = useState("");
    const [message, setMessage] = useState("");
    const [isVerifying, setIsVerifying] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();
    const { email } = location.state || {};

    const handleVerifyCode = async () => {
        if (!code.trim()) {
            setMessage("Please enter the code sent to your email.");
            return;
        }

        setIsVerifying(true);
        setMessage("");

        try {
            const response = await fetch("http://localhost:8000/verify-code", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, code }),
            });

            const data = await response.json();
            if (response.ok) {
                setMessage("Code verified! You can now reset your password.");
                setTimeout(() => navigate("/reset-password-form", { state: { email } }), 1500);
            } else {
                setMessage(data.error || "Invalid or expired code.");
            }
        } catch {
            setMessage("Server error. Please try again later.");
        } finally {
            setIsVerifying(false);
        }
    };

    const handleGoBack = () => {
        navigate("/reset-password", { replace: true });
    };

    const handleKeyPress = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter') {
            handleVerifyCode();
        }
    };

    return (
        <Box
            sx={{
                // Style from Registration outer Box
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                position: "absolute",
                inset: 0,
                background: "linear-gradient(135deg, #1A2027 0%, #171A21 100%)",
                color: "#e0e0e0",
                fontFamily: "Roboto, sans-serif",
                p: 3,
            }}
        >
            <Box
                sx={{
                    // Style from Registration inner Box
                    p: { xs: 4, md: 5 },
                    bgcolor: "rgba(30, 32, 35, 0.98)",
                    backdropFilter: "blur(8px)",
                    borderRadius: 3,
                    boxShadow: "0 8px 30px rgba(0,0,0,0.7)",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 3,
                    width: "100%",
                    maxWidth: 400,
                    border: "1px solid rgba(255, 255, 255, 0.05)",
                    overflowY: 'auto',
                    '& *': {
                        outline: 'none !important',
                    },
                    '& *:focus': {
                        outline: 'none !important',
                    },
                }}
            >
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, mb: 1 }}>
                    {/* Style from Registration Icon */}
                    <VerifiedUserIcon sx={{ fontSize: 60, color: "#1a73e8" }} />

                    <Typography
                        variant="h5"
                        component="h1"
                        sx={{
                            // Style from Registration h5
                            color: '#e0e0e0',
                            fontWeight: 600,
                            letterSpacing: 0.5,
                            textTransform: 'uppercase',
                        }}
                    >
                        Verify Your Code
                    </Typography>

                    <Typography variant="body2" sx={{
                        // Style from Registration body2
                        color: '#8e8e8e',
                        mt: -1,
                        textAlign: "center"
                    }}>
                        Enter the 4-digit code sent to <strong>{email}</strong>.
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="outlined"
                    label="4-Digit Code"
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{
                        // Style from Registration TextField
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
                            '&.Mui-focused fieldset': { // Corrected selector
                                borderColor: '#1a73e8',
                                borderWidth: '2px',
                            },
                        },
                        // Merged original input styles
                        '& .MuiInputBase-input': {
                            color: '#e0e0e0',
                            textAlign: "center",
                            // fontSize: "1.5rem", // <-- Removed this line
                            letterSpacing: "0.5rem",
                        },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                    }}
                />
                <Button
                    fullWidth
                    variant="contained"
                    onClick={handleVerifyCode}
                    disabled={isVerifying}
                    sx={{
                        // Style from Registration Button
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
                    {isVerifying ? "Verifying..." : "Verify Code"}
                </Button>

                {message && (
                    <Typography
                        variant="body2"
                        sx={{
                            // Original logic, styled to fit
                            color: message.includes("verified") ? "#4caf50" : "#f44336",
                            textAlign: "center",
                        }}
                    >
                        {message}
                    </Typography>
                )}


                <Button
                    variant="text"
                    onClick={handleGoBack}
                    startIcon={<ArrowBackIcon />} // Icon added back
                    sx={{
                        // Style from Registration Link
                        color: '#8e8e8e',
                        textTransform: 'none',
                        '&:hover': {
                            color: '#e0e0e0',
                            textDecoration: 'none',
                            bgcolor: 'transparent' // ensure no button bg on hover
                        },
                    }}
                >
                    Back
                </Button>
            </Box>
        </Box>
    );
}