import { Box, Typography, List, ListItem, ListItemButton, ListItemText, ListItemIcon, Button } from '@mui/material';
import LiveHelpIcon from '@mui/icons-material/LiveHelp';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import VpnKeyIcon from '@mui/icons-material/VpnKey';
import HourglassBottomIcon from '@mui/icons-material/HourglassBottom';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import {useNavigate} from "react-router-dom";
import SettingsIcon from "@mui/icons-material/Settings";
import {useEffect} from "react";
const Choice = () => {


    const navigate = useNavigate()

    const menuItemsLoggedIn = [
        {text: "Chat", icon: <LiveHelpIcon />, value: "chat" },
        {text: "Temporary Chat", icon: <HourglassBottomIcon />, value: "chatTemp" },
        {text: "PDF Manager", icon: <PictureAsPdfIcon />, value: "checkDB" },
        {text: "Enter API Key", icon: <VpnKeyIcon /> , value: "apiKey" },
        {text: "Settings", icon:<SettingsIcon /> , value: "settings"},
        {text: "Logout", icon: <ExitToAppIcon />, value: "logout" }
    ];

    const guestMenuItems = [
        {text: "Chat", icon: <LiveHelpIcon />, value: "chat" },
        {text: "Exit", icon: <ExitToAppIcon />, value: "logout" }
    ]

    // @ts-ignore
    function handleMenu(value) {
        switch (value) {
            case "chat": navigate("/ChatBot"); break;
            case "chatTemp": navigate("/chatbotTemp"); break;
            case "insertPDF": navigate("/uploadPDF"); break;
            case "apiKey": navigate("/StoreKey"); break;
            case "checkDB": navigate("/PDFManager"); break;
            case "settings": navigate("/Settings"); break;
            case "logout":

                localStorage.clear()
                navigate("/Login");

                break;
        }
    }

    useEffect(() => {

    }, []);



    function menuItemLogic() {
        if (localStorage.getItem("loggedIn") === "true")
            return menuItemsLoggedIn;

        return guestMenuItems;
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
                <Typography
                    variant="h5"
                    component="h1"
                    sx={{
                        color: '#e0e0e0',
                        fontWeight: 600,
                        letterSpacing: 0.5,
                        mb: 1,
                        width: 'auto',
                        textTransform: 'uppercase',
                    }}
                >
                    Main Menu
                </Typography>
                <List sx={{ width: '100%', p: 0 }}>
                    {menuItemLogic().map((item, index) => (
                        <ListItem key={index} disablePadding sx={{ mb: 1.5 }}>
                            <ListItemButton
                                onClick={() => handleMenu(item.value)}
                                sx={{
                                    py: 1.25,
                                    px: 2,
                                    borderRadius: 1,
                                    transition: 'all 0.3s',
                                    bgcolor: 'transparent',
                                    '&:hover': {
                                        bgcolor: '#282a2e',
                                        transform: 'translateY(-1px)',
                                        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)',
                                    },
                                }}
                            >
                                <ListItemIcon
                                    sx={{
                                        color: '#1a73e8',
                                        minWidth: 40
                                    }}
                                >
                                    {item.icon}
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Typography sx={{
                                            color: '#e0e0e0',
                                            fontWeight: 500,
                                        }}>
                                            {item.text}
                                        </Typography>
                                    }
                                />
                            </ListItemButton>
                        </ListItem>
                    ))}
                </List>

                {/* Admin Panel Button */}
                {localStorage.getItem("role") === "admin" && (
                    <Button
                        fullWidth
                        variant="contained"
                        onClick={() => navigate("/admin")}
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
                        Admin Panel
                    </Button>
                )}

                <Typography
                    variant="caption"
                    sx={{
                        color: '#8e8e8e',
                        mt: 1
                    }}
                >
                </Typography>
            </Box>
        </Box>
    );
};

export default Choice;
