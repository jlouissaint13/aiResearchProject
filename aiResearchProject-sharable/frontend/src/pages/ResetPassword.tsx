import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Box, Typography, TextField, Button } from "@mui/material";
import LockResetIcon from "@mui/icons-material/LockReset";

export default function ResetPassword() {
    const [email, setEmail] = useState("");
    const [message, setMessage] = useState("");
    const [isSending, setIsSending] = useState(false);
    const navigate = useNavigate();

    const handleSendResetLink = async () => {
        if (!email.trim()) {
            setMessage("Please enter your email.");
            return;
        }

        setIsSending(true);
        setMessage("");

        try {
            const response = await fetch("http://localhost:8000/forgot-password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email }),
            });

            const data = await response.json();

            if (response.ok) {
                navigate("/verify-code", { state: { email } });
            } else {
                setMessage(data.error || "Unable to send reset code.");
            }
        } catch {
            setMessage("Server error. Please try again later.");
        } finally {
            setIsSending(false);
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
                }}
            >
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, mb: 1 }}>
                    {/* Style from Registration Icon */}
                    <LockResetIcon sx={{ fontSize: 60, color: "#1a73e8" }} />
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
                        Forgot Password?
                    </Typography>
                    <Typography variant="body2" sx={{
                        // Style from Registration body2
                        color: '#8e8e8e',
                        mt: -1
                    }}>
                        Enter your email to receive a 4-digit reset code.
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="outlined" // Changed from "filled"
                    label="Email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
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

                <Button
                    fullWidth
                    variant="contained"
                    onClick={handleSendResetLink}
                    disabled={isSending}
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
                    {isSending ? "Sending..." : "Send Reset Code"}
                </Button>

                {message && (
                    <Typography
                        variant="body2"
                        sx={{
                            // Original logic, slightly styled to fit
                            color: message.includes("sent") ? "#4caf50" : "#f44336", // Uses red from Registration error
                        }}
                    >
                        {message}
                    </Typography>
                )}


                <Button
                    variant="text"
                    onClick={() => navigate("/login")}
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
                    ← Back to Login
                </Button>
            </Box>
        </Box>
    );
}