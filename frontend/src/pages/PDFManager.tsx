import { useState, useRef } from 'react';
import { Box, Typography, Button, IconButton, List, ListItem, ListItemText, ListItemIcon, ListItemButton, TextField } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import DeleteIcon from '@mui/icons-material/Delete';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import {useNavigate} from "react-router-dom";
import * as pdfjsLib from 'pdfjs-dist/webpack';


const initialPdfs = [
    {  name: 'test_a.pdf' },
    {  name: 'test_b.pdf' },
    {  name: 'test_3.pdf' },
    {  name: 'test_4.pdf' },
];

const PdfManager = () => {
    const [pdfs, setPdfs] = useState(initialPdfs);
    const [searchTerm, setSearchTerm] = useState('');
    const navigate = useNavigate();
    
    
    const fileInputRef = useRef(null);

    const handleBack = () => {
        navigate("/Choice")
    };
    
    
    const handleAddPdf = () => {
        fileInputRef.current.click();
    };
    
  async function extractMetaData(file) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf= await pdfjsLib.getdocument({data: arrayBuffer}).promise
        const metadata = await pdf.getMetadata();
        
        
        return metadata
    }



    async function handleFileChange(event) {
        const files = event.target.files;

        if (files.length > 0) {
            const newFile = files[0];
            const metadata = await extractMetaData(newFile);  
            console.log("Title:", metadata.info.Title);

            const newPdf = { name: newFile.name };
            setPdfs(prevPdfs => [...prevPdfs, newPdf]);

            alert(`PDF selected and added: ${newFile.name}`);

            event.target.value = null;
        }
    }

    const handleDeletePdf = (id) => {
        console.log(`Deleting PDF with ID: ${id}`);
        setPdfs(prevPdfs => prevPdfs.filter(pdf => pdf.id !== id));
    };

    const filteredPdfs = pdfs.filter(pdf =>
        pdf.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

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
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".pdf"
                style={{ display: 'none' }}
            />
            <Box
                sx={{
                    position: 'absolute',
                    top: 24,
                    left: 24,
                    zIndex: 10,
                }}
            >
                <Button
                    onClick={handleBack}
                    variant="text"
                    startIcon={<ArrowBackIcon />}
                    sx={{
                        color: '#e0e0e0',
                        textTransform: 'none',
                        '&:hover': {
                            bgcolor: 'rgba(255, 255, 255, 0.08)',
                        },
                    }}
                >
                    Back to Menu
                </Button>
            </Box>

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
                    maxWidth: 600,
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    maxHeight: '90vh',
                }}
            >
                <Typography variant="h5" component="h1" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>
                    Manage Documents
                </Typography>
                <Typography variant="body2" sx={{ color: '#8e8e8e', textAlign: 'center' }}>
                    These documents are available for context-aware chat.(Max 10MB)
                </Typography>

                <TextField
                    fullWidth
                    variant="filled"
                    label="Search documents..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    sx={{
                        '& .MuiFilledInput-root': {
                            borderRadius: 2,
                            bgcolor: '#3e4042',
                            '&:hover': { bgcolor: '#424549' },
                            '&.Mui-focused': { bgcolor: '#3e4042' },
                        },
                        '& .MuiInputBase-input': { color: '#e0e0e0' },
                        '& .MuiInputLabel-root': { color: '#8e8e8e' },
                        '& .MuiInputLabel-root.Mui-focused': { color: '#1a73e8' },
                    }}
                />

                <Button
                    fullWidth
                    variant="contained"
                    onClick={handleAddPdf}
                    startIcon={<AddCircleIcon />}
                    sx={{
                        p: 1,
                        borderRadius: 2,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        textTransform: 'none',
                        '&:hover': {
                            bgcolor: '#1565c0',
                            boxShadow: '0px 4px 15px rgba(26, 115, 232, 0.4)',
                        },
                    }}
                >
                    Add New PDF
                </Button>


                <Box sx={{ width: '100%', overflowY: 'auto', flexGrow: 1, mt: 2 }}>
                    {filteredPdfs.length === 0 ? (
                        <Typography sx={{ color: '#8e8e8e', textAlign: 'center', mt: 4 }}>
                            {searchTerm ? `No documents found matching "${searchTerm}".` : 'No PDF documents currently uploaded.'}
                        </Typography>
                    ) : (
                        <List sx={{ width: '100%' }}>
                            {filteredPdfs.map((pdf) => (
                                <ListItem
                                    key={pdf.id}
                                    disablePadding
                                    secondaryAction={
                                        <IconButton edge="end" aria-label="delete" onClick={() => handleDeletePdf(pdf.id)} sx={{ color: '#f44336' }}>
                                            <DeleteIcon />
                                        </IconButton>
                                    }
                                    sx={{
                                        borderBottom: '1px solid #3e4042',
                                        '&:last-child': { borderBottom: 'none' },
                                    }}
                                >
                                    <ListItemButton sx={{ py: 1.5, px: 1, borderRadius: 2, '&:hover': { bgcolor: '#424549' } }}>
                                        <ListItemIcon sx={{ color: '#e0e0e0' }}>
                                            <PictureAsPdfIcon />
                                        </ListItemIcon>
                                        <ListItemText primary={
                                            <Typography sx={{
                                                color: '#e0e0e0',
                                                whiteSpace: 'nowrap',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis'
                                            }}>
                                                {pdf.name}
                                            </Typography>
                                        } />
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    )}
                </Box>
            </Box>
        </Box>
    );
};

export default PdfManager;