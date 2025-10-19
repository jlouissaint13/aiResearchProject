import {useEffect, useRef, useState} from 'react';
import {
    Box, Button, IconButton, List, ListItem, ListItemButton, ListItemIcon, ListItemText, TextField, Typography
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import DeleteIcon from '@mui/icons-material/Delete';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import {useNavigate} from "react-router-dom";
import axios from 'axios';

const initialPdfs = [
    { id: 101, name: 'a', filePath: 'uploads/test_a.pdf' },
   
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
    

    function extractFileInfo(file: File) {
        const metadata = {
            fileName: file.name || '',
            filePath: (window as any).fileAPI.getFilePath(file),
            fileSize: file.size || 0,
           
        };

        return metadata;
    }



    async function handleFileChange(event) {
        const files = event.target.files;
        if (files.length > 0) {
            const newFile = files[0];
            
            const fileInfo = extractFileInfo(newFile); 
            const newPdf = { name: newFile.name };
            const fileData = {
                file_path: fileInfo.filePath,
                file_name: newFile.name,
                user_id: localStorage.getItem("userID")
            }
            try {
                const response = await axios.post('http://localhost:8000/pdf_manager/insert_pdf', fileData);
                
                
                if (response.status === 200) {
                    alert("PDF added successfully")
                    await retrieveAllPDFS();
                    return
                }
                
            }catch (error) {
               if (axios.isAxiosError(error)) {
                   const status = error.response?.status
                   
                   if (status === 409) {
                       alert("PDF has already been added")
                   }
                   
               }
                   
            }
            
            event.target.value = null;
        }
    }

    useEffect(() => {
       retrieveAllPDFS()
    }, []);
    
    async function retrieveAllPDFS() {
        const data = {
            user_id: localStorage.getItem('userID')
        }
        try {
            const response = await axios.post('http://localhost:8000/pdf_manager/retrieve_all_pdfs',data)
            
            const pdfData = response.data
            
            const formattedPDFs = pdfData.map((pdf: { pdf_id: any; pdf_name: string; file_path: string; }) =>({
                id: pdf.pdf_id,
                name: pdf.pdf_name,
                filePath: pdf.file_path
            }));

            setPdfs(formattedPDFs)
           

           
            
        }catch (error) {
            alert(error)
        }
        
    }
    
    
    const handleDeletePdf = async (name,file_path) => {

        const isConfirmed = window.confirm(
            `Are you sure you want to delete the PDF titled: "${name}"? This action cannot be undone.`
        );
        
        if (!isConfirmed) {
            return
        }
        const dataInfo = {
            user_id : localStorage.getItem("userID"),
            file_path: file_path
        };
        try {
            const response = await axios.delete('http://localhost:8000/pdf_manager/delete_pdf', {
                data: dataInfo
            });
            
            
            if (response.status === 200) {
                await retrieveAllPDFS()
                
            }
            
            
            
        }catch (error) {
            
        }
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
                                        <IconButton edge="end" aria-label="delete" onClick={() => handleDeletePdf(pdf.name,pdf.filePath)} sx={{ color: '#f44336' }}>
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