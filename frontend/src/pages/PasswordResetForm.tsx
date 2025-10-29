import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Box, Typography, TextField, Button } from "@mui/material";
import LockIcon from "@mui/icons-material/Lock";
import ArrowBackIcon from "@mui/icons-material/ArrowBack"; // Added back

export default function ResetPasswordForm() {
    const [newPassword, setNewPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [message, setMessage] = useState("");
    const [isSaving, setIsSaving] = useState(false);
    const [passwordError, setPasswordError] = useState(false);
    const [confirmError, setConfirmError] = useState(false);
    const [helperText, setHelperText] = useState("");

    const navigate = useNavigate();
    const location = useLocation();
    const { email } = location.state || {};

    const handleSavePassword = async () => {
        // Reset errors
        setPasswordError(false);
        setConfirmError(false);
        setHelperText("");
        setMessage("");

        if (!newPassword.trim() || !confirmPassword.trim()) {
            setMessage("Please fill in both fields.");
            setPasswordError(!newPassword.trim());
            setConfirmError(!confirmPassword.trim());
            setHelperText("Please fill in both fields.");
            return;
        }

        if (newPassword !== confirmPassword) {
            setMessage("Passwords do not match.");
            setPasswordError(true);
            setConfirmError(true);
            setHelperText("Passwords do not match.");
            return;
        }

        setIsSaving(true);

        try {
            const response = await fetch("http://localhost:8000/reset-password", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email,
                    new_password: newPassword,
                    confirm_password: confirmPassword,
                }),
            });

            const data = await response.json();

            if (response.ok) {
                setMessage("Password successfully updated! Redirecting to login...");
                setTimeout(() => navigate("/login"), 2000);
            } else {
                setMessage(data.error || "Failed to reset password.");
            }
        } catch {
            setMessage("Server error. Please try again later.");
        } finally {
            setIsSaving(false);
        }
    };

    const handleGoBack = () => {
        navigate("/verify-code", { state: { email }, replace: true });
    };

    const handleKeyPress = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter') {
            handleSavePassword();
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
                    <LockIcon sx={{ fontSize: 60, color: "#1a73e8" }} />

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
                        Reset Your Password
                    </Typography>

                    <Typography variant="body2" sx={{
                        // Style from Registration body2
                        color: '#8e8e8e',
                        mt: -1,
                        textAlign: "center"
                    }}>
                        Enter your new password below.
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="outlined" // Changed from "filled"
                    label="New Password"
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    onKeyPress={handleKeyPress}
                    error={passwordError}
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

                <TextField
                    fullWidth
                    variant="outlined" // Changed from "filled"
                    label="Confirm Password"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    onKeyPress={handleKeyPress}
                    error={confirmError}
                    helperText={helperText} // Helper text only on the second field
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
                        '& .MMuiInputBase-input': { color: '#e0e0eG' }, // Typo fixed
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                        '& .MuiFormHelperText-root': { // Style from Registration helperText
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
                    onClick={handleSavePassword}
                    disabled={isSaving}
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
                    {isSaving ? "Saving..." : "Save Password"}
                </Button>

                {/* Show server messages only if not a form validation error */}
                {message && !helperText && (
                    <Typography
                        variant="body2"
                        sx={{
                            color: message.includes("successfully") ? "#4caf50" : "#f44336",
                            textAlign: "center",
                        }}
                    >
                        {message}
                    </Typography>
                )}

                <Button
                    variant="text"
                    onClick={handleGoBack}
                    startIcon={<ArrowBackIcon />} // Added back
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