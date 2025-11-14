import {useEffect, useRef, useState} from 'react';
import {
    Box, Button, IconButton, List, ListItem, ListItemButton, ListItemIcon, ListItemText, TextField, Typography, Divider,
    LinearProgress 
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import DeleteIcon from '@mui/icons-material/Delete';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import {useNavigate} from "react-router-dom";
import axios from 'axios';

const initialPdfs = [
    { id: 101, name: 'Sample Document A.pdf', filePath: 'uploads/test_a.pdf' },

];

const PdfManager = () => {
    const [pdfs, setPdfs] = useState(initialPdfs);
    const [searchTerm, setSearchTerm] = useState('');
    const [isLoading, setIsLoading] = useState(false); 
    const navigate = useNavigate();

    const fileInputRef = useRef(null);

    const handleBack = () => {
        navigate("/Choice")
    };


    const handleAddPdf = () => {
        // @ts-ignore
        fileInputRef.current.click();
    };


    // @ts-ignore
    function extractFileInfo(file) {
        // @ts-ignore

        const metadata = {
            fileName: file.name || '',
            filePath: (window).fileAPI?.getFilePath(file),
            fileSize: file.size || 0,
        };

        return metadata;
    }



    // @ts-ignore
    async function handleFileChange(event) {
        const files = event.target.files;
        if (files.length > 0) {
            setIsLoading(true); 
            const newFile = files[0];

            const fileInfo = extractFileInfo(newFile);
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
                }

            }catch (error) {
                if (axios.isAxiosError(error)) {
                    const status = error.response?.status

                    if (status === 409) {
                        alert("PDF has already been added")
                    }
                }
            } finally {
                setIsLoading(false); 
                event.target.value = null; 
            }
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

            // @ts-ignore
            const formattedPDFs = pdfData.map((pdf) =>({
                id: pdf.pdf_id,
                name: pdf.pdf_name,
                filePath: pdf.file_path
            }));

            setPdfs(formattedPDFs)

        }catch (error) {
            alert(error)
        }

    }


    // @ts-ignore
    const handleDeletePdf = async (name,file_path) => {

        
        if (isLoading) return; 

        const isConfirmed = window.confirm(
            `Are you sure you want to delete the PDF titled: "${name}"? This action cannot be undone.`
        );

        if (!isConfirmed) {
            return
        }

        setIsLoading(true); 
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
           
        } finally {
            setIsLoading(false); 
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
                disabled={isLoading} 
            />
            <Box
                sx={{
                    position: 'absolute',
                    top: { xs: 16, md: 20 },
                    left: { xs: 16, md: 20 },
                    zIndex: 10,
                }}
            >
                <Button
                    onClick={handleBack}
                    variant="text"
                    startIcon={<ArrowBackIcon />}
                    disabled={isLoading} 
                    sx={{
                        color: '#a0a0a0',
                        textTransform: 'none',
                        fontSize: '0.85rem',
                        p: 0.5,
                        borderRadius: 1,
                        '&:hover': {
                            bgcolor: 'rgba(255, 255, 255, 0.05)',
                            color: '#e0e0e0',
                        },
                    }}
                >
                    Back to Menu
                </Button>
            </Box>

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
                    gap: 2,
                    width: '100%',
                    maxWidth: 500,
                    maxHeight: '90vh',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    position: 'relative', 
                    overflow: 'hidden',   
                }}
            >
                {isLoading && (
                    <LinearProgress
                        sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            height: '4px',
                            backgroundColor: 'rgba(26, 115, 232, 0.3)',
                            '& .MuiLinearProgress-bar': {
                                backgroundColor: '#1a73e8',
                            },
                        }}
                    />
                )}

                <Box sx={{ width: '100%', textAlign: 'center', mb: 1 }}>
                    <Typography
                        variant="h5"
                        component="h1"
                        sx={{ color: '#e0e0e0', fontWeight: 600 }}
                    >
                        Document Manager
                    </Typography>
                    <Typography
                        variant="caption"
                        sx={{ color: '#8e8e8e', mt: 0.5 }}
                    >
                        (Max 10MB)
                    </Typography>
                </Box>

                <TextField
                    fullWidth
                    variant="outlined"
                    label="Search documents..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    disabled={isLoading} 
                    sx={{
                        mt: 1,
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
                    onClick={handleAddPdf}
                    startIcon={<AddCircleIcon />}
                    disabled={isLoading} 
                    sx={{
                        p: 1.25,
                        borderRadius: 1,
                        bgcolor: '#1a73e8',
                        color: '#fff',
                        fontWeight: 600,
                        textTransform: 'none',
                        boxShadow: 'none',
                        '&:hover': {
                            bgcolor: '#1565c0',
                            boxShadow: '0px 2px 8px rgba(26, 115, 232, 0.5)',
                        },
                    }}
                >
                    Add New PDF Document
                </Button>

                <Divider sx={{ width: '100%', bgcolor: 'rgba(255, 255, 255, 0.08)' }} />


                <Box sx={{ width: '100%', overflowY: 'auto', flexGrow: 1, mt: 1 }}>
                    {filteredPdfs.length === 0 ? (
                        <Typography sx={{ color: '#8e8e8e', textAlign: 'center', mt: 4, fontStyle: 'italic' }}>
                            {searchTerm
                                ? `No documents found matching "${searchTerm}"...`
                                : 'No PDF documents currently uploaded. '
                            }
                        </Typography>
                    ) : (
                        <List sx={{
                            width: '100%',
                            p: 0,
                        }}>
                            {filteredPdfs.map((pdf) => (
                                <ListItem
                                    key={pdf.id}
                                    disablePadding
                                    secondaryAction={
                                        <IconButton
                                            edge="end"
                                            aria-label="delete"
                                            onClick={() => handleDeletePdf(pdf.name, pdf.filePath)}
                                            disableRipple
                                            disableFocusRipple
                                            disableTouchRipple
                                            disabled={isLoading} 
                                            sx={{
                                                color: '#f44336',
                                                p: 1,
                                                '&:hover': {
                                                    bgcolor: 'rgba(244, 67, 54, 0.15)',
                                                },
                                                '&:focus, &.Mui-focusVisible': {
                                                    outline: 'none',
                                                    boxShadow: 'none',
                                                    bgcolor: 'transparent',
                                                },

                                                '&:active': {
                                                    bgcolor: 'transparent',
                                                    boxShadow: 'none',
                                                },
                                                '& .MuiTouchRipple-root': {
                                                    display: 'none',
                                                },
                                            }}
                                        >
                                            <DeleteIcon />
                                        </IconButton>
                                    }
                                    sx={{
                                        bgcolor: 'rgba(62, 64, 66, 0.05)',
                                        borderBottom: '1px solid #3e4042',
                                        '&:first-of-type': { borderTopLeftRadius: 3, borderTopRightRadius: 3},
                                        '&:last-child': { borderBottom: 'none', borderBottomLeftRadius: 3, borderBottomRightRadius: 3 },

                                    }}
                                >
                                    <ListItemButton
                                        disabled={isLoading} 
                                        sx={{
                                            py: 1.2,
                                            px: 1.5,
                                            '&:hover': { bgcolor: '#424549' }
                                        }}
                                    >
                                        <ListItemIcon sx={{ color: '#1a73e8', minWidth: 40 }}>
                                            <PictureAsPdfIcon />
                                        </ListItemIcon>
                                        <ListItemText primary={
                                            <Typography sx={{
                                                color: '#e0e0e0',
                                                whiteSpace: 'nowrap',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                fontWeight: 500
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