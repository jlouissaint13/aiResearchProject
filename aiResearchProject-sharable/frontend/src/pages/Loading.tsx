import {useEffect} from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import {useNavigate} from 'react-router-dom';




const Loading = () => {

    const navigate = useNavigate();
    useEffect(()=> {
        setTimeout(loadLogin,2000)
    },[]);



//simulated loading for now
    function loadLogin() {
        navigate("/Login")
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
                p: 3,
                gap: 3,
            }}
        >
            <CircularProgress sx={{ color: '#1a73e8' }} size={60} />
            <Typography variant="h6" component="p" sx={{ color: '#e0e0e0' }}>
                Initializing Marie...
            </Typography>
        </Box>
    );
};

export default Loading;