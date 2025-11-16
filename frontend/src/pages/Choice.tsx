import { Box, Typography, List, ListItem, ListItemButton, ListItemText, ListItemIcon} from '@mui/material';
import LiveHelpIcon from '@mui/icons-material/LiveHelp';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import VpnKeyIcon from '@mui/icons-material/VpnKey';
import HourglassBottomIcon from '@mui/icons-material/HourglassBottom';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import {useNavigate} from "react-router-dom";
import SettingsIcon from "@mui/icons-material/Settings";
import BarChartIcon from '@mui/icons-material/BarChart';
import {useEffect, useState} from "react";
import axios from "axios";
const Choice = () => {


    const navigate = useNavigate()
    const [isDataVisualDisabled,setDataVisualDisabled] = useState(true);

    const menuItemsLoggedIn = [
        {text: "Chat", icon: <LiveHelpIcon />, value: "chat", disabled: false },
        {text: "Temporary Chat", icon: <HourglassBottomIcon />, value: "chatTemp", disabled: false },
        {text: "Data Visualization (BETA)", icon: <BarChartIcon />, value: "dataVisual", disabled: isDataVisualDisabled },
        {text: "PDF Manager", icon: <PictureAsPdfIcon />, value: "checkDB", disabled: false },
        {text: "Enter API Key", icon: <VpnKeyIcon />, value: "apiKey", disabled: false },
        {text: "Settings", icon:<SettingsIcon />, value: "settings", disabled: false },
        {text: "Logout", icon: <ExitToAppIcon />, value: "logout", disabled: false }
    ];

    const guestMenuItems = [
        {text: "Chat", icon: <LiveHelpIcon />, value: "chat", disabled: false },
        {text: "Exit", icon: <ExitToAppIcon />, value: "logout", disabled: false }
    ];

    // @ts-ignore
    function handleMenu(value) {
        switch (value) {
            case "chat": navigate("/ChatBot"); break;
            case "chatTemp": navigate("/chatbotTemp"); break;
            case "dataVisual": navigate("/dataVisual"); break;
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
        retrieveUserSettings()
    }, []);


    function menuItemLogic() {
        if (localStorage.getItem("loggedIn") === "true")
            return menuItemsLoggedIn;

        return guestMenuItems;
    }


    async function retrieveUserSettings() {
        const userId : string | null = localStorage.getItem("userID");


        try {
            const response = await axios.post('http://localhost:8000/model_settings/data_visualization_allowed', {
                user_id: userId
            });

            if (response.status === 200 && response.data) {
                    setDataVisualDisabled(false)
                    

            } else if (response.status === 404) {
                return
            }
        } catch (error : any) {
            if (error.response && error.response.status === 404) {

            } else {
                console.error("An error occurred while retrieving user settings:", error);
            }
        }
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
                                disabled={item.disabled}
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