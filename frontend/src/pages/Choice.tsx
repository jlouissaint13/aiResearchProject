
import { Box, Typography, List, ListItem, ListItemButton, ListItemText, ListItemIcon} from '@mui/material';
import LiveHelpIcon from '@mui/icons-material/LiveHelp';
import SearchIcon from '@mui/icons-material/Search';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import VpnKeyIcon from '@mui/icons-material/VpnKey';

import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import {useNavigate} from "react-router-dom";
import SettingsIcon from "@mui/icons-material/Settings";
import {useEffect} from "react";
const Choice = () => {




    const navigate = useNavigate()

    const menuItemsLoggedIn = [
        {text: "Chat", icon: <LiveHelpIcon />, value: "chat" },
        {text: "Quick Search", icon: <SearchIcon /> , value: "quickSearch" },
        {text: "PDF Manager", icon: <PictureAsPdfIcon />, value: "checkDB" },
        {text: "Enter API Key", icon: <VpnKeyIcon /> , value: "apiKey" },
        {text: "Settings", icon:<SettingsIcon /> , value: "settings"},
        {text: "Logout", icon: <ExitToAppIcon />, value: "logout" }
    ];
    
    const guestMenuItems = [
        {text: "Chat", icon: <LiveHelpIcon />, value: "chat" },
        {text: "Quick Search", icon: <SearchIcon /> , value: "quickSearch" },
        {text: "Exit", icon: <ExitToAppIcon />, value: "logout" }
    ]

    function handleMenu(value:string) {
        switch (value) {
            case "chat": navigate("/ChatBot"); break;
            case "quickSearch": break;
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



    function menuItemLogic(): any {
        if (localStorage.getItem("loggedIn") === "true")
            return menuItemsLoggedIn;
        
        return guestMenuItems;
    }
    
    

    // @ts-ignore
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
                    gap: 3,
                    width: '100%',
                    maxWidth: 450,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
            >
                <Typography variant="h5" component="h1" sx={{ color: '#e0e0e0', fontWeight: 'bold', letterSpacing: 1, mb: 1 }}>
                    Please make a selection:
                </Typography>
                <List sx={{ width: '100%' }}>
                    {menuItemLogic().map((item, index) => (
                        <ListItem key={index} disablePadding>
                            <ListItemButton
                                onClick={() => handleMenu(item.value)}
                                sx={{
                                    py: 1.5,
                                    px: 2,
                                    borderRadius: 2,
                                    transition: 'background-color 0.3s ease-in-out',
                                    '&:hover': {
                                        bgcolor: '#424549',
                                    },
                                }}
                            >
                                <ListItemIcon sx={{ color: '#88aaff' }}>
                                    {item.icon}
                                </ListItemIcon>
                                <ListItemText primary={item.text} sx={{ color: '#e0e0e0' }} />
                            </ListItemButton>
                        </ListItem>
                    ))}
                </List>
            </Box>
        </Box>
    );
};

export default Choice;
