import { useState, useEffect, useRef } from 'react';
import { Box, Button, Typography, TextField, IconButton, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import SettingsIcon from '@mui/icons-material/Settings';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import CreateIcon from '@mui/icons-material/Create';
const LOADING_MESSAGES = [
    "Consulting the data...",
    "Synthesizing your request...",
    "Formulating a response...",
    "Almost there! Just a moment...",
    "Finalizing everything for you..."
];

const AVAILABLE_MODELS = [
    { name: "Gemini 2.5 Flash", id: "gemini-2.5" },
    {name: "LLAMA3.2B", id: "llama3.2b"}
   
];


const ChatBot = () => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [currentLoadingMessage, setCurrentLoadingMessage] = useState('');
    const messagesEndRef = useRef(null);
    // @ts-ignore
    const [valueLoadingMessage,setValueLoadingMessage] = useState<number>(0)
    const valueRef = useRef(0)
    const navigate = useNavigate();
    const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0].id);

    interface RecentChat {
        title: string;
        conversationID: string;
        lastModified: string
    }

    interface ChatMessage {
        id: string;
        content: string;
        sender: 'user' | 'model';
    }

// @ts-ignore
    function sortChats(chats) {
        return [...chats].sort((a, b) =>
            +new Date(b.lastModified) - +new Date(a.lastModified)
        );
    }
    const [recentChats, setRecentChats] = useState<RecentChat[]>([]);

    useEffect(() => {
        getConversations()
    }, []);

    const intervalRef = useRef(null);

    useEffect(() => {
        if (isLoading) {
            loadingMessagesControl(valueRef.current);
            // @ts-ignore
            intervalRef.current = setInterval(() => {
                loadingMessagesControl(valueRef.current)
            }, 5000)
        }
        else {
            setCurrentLoadingMessage('');
            valueRef.current = 0;
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [isLoading]);


    function loadingMessagesControl(i: number) : void {
        if (i < LOADING_MESSAGES.length) {
            setCurrentLoadingMessage(LOADING_MESSAGES[i]);
            const next : number = i + 1;
            setValueLoadingMessage(next);
            valueRef.current = next;
        }
    }


    async function getConversations() {
        const data = {
            user_id : localStorage.getItem("userID")
        }

        try {
            const response = await fetch('http://localhost:8000/conversation/get_conversations_by_id', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            if (response.status === 200) {
                const conversations = await response.json();
                const newChats = [];

                for (let i = 0; i < conversations.length; i++) {
                    newChats.push({
                        title: conversations[i][4],
                        conversationID: conversations[i][0],
                        lastModified: conversations[i][5]
                    });
                }

                setRecentChats(sortChats(newChats));
            }

        } catch (error) {
            console.error("Error getting conversations:", error);
            return;
        }
    }

    async function handleChatClick(conversationID:string) {
        setIsSidebarOpen(false);
        setMessages([]);

        sessionStorage.setItem("conversationID", conversationID);

        const data = {
            conversation_id: conversationID,
            user_id: localStorage.getItem("userID")
        };

        try {
            const response = await fetch('http://localhost:8000/message/get_messages_by_conversation_id', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });

            if (response.status === 200) {
                const fetchedMessages = await response.json();

                const formattedMessages = fetchedMessages.map(function(msg: { content: string; role: string; message_id: any; }) {
                    return {
                        content: msg.content,
                        sender: msg.role,
                        id: msg.message_id,
                    };
                });

                setMessages(formattedMessages);
            }

        } catch (error) {
            console.error("Error handling chat click:", error);
        }
    }
    async function handleDeleteChat(conversationID: string) {
        const isConfirmed = window.confirm("Are you sure you want to delete this conversation? This action cannot be undone.");

        if (!isConfirmed) return;

        setIsLoading(true);

        try {
            const data = {
                conversation_id: conversationID,
                user_id : localStorage.getItem("userID")
            }

            const response = await fetch('http://localhost:8000/conversation/delete_conversation', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });

            if (response.status === 200) {
                setRecentChats(prevChats =>
                    prevChats.filter(chat => chat.conversationID !== conversationID)
                );

                if (sessionStorage.getItem("conversationID") === conversationID) {
                    sessionStorage.removeItem("conversationID");
                    setMessages([]);
                }
            } else {
                alert("Failed to delete chat.");
            }

        } catch (error) {
            alert("Error deleting chat: " + error);
        } finally {
            setIsLoading(false);
        }
    }

    const handleRightClick = (e: React.MouseEvent, conversationID: string) => {
        if (isLoading) return;
        e.preventDefault();

        const shouldDelete = true;

        if (shouldDelete) {
            handleDeleteChat(conversationID);
        }
    }

    async function createChat(title : string) {
        let conversationID = uuidv4();
        sessionStorage.setItem("conversationID",conversationID);
        sessionStorage.setItem("mostRecentID",conversationID)
        const currentTime = new Date().toISOString();
        const newChatDataForBackend = {
            user_id: localStorage.getItem("userID"),
            title : title,
            conversation_id: conversationID,
        }

        try {
            const response = await fetch('http://localhost:8000/conversation/receive', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(newChatDataForBackend),
            });

            if (response.status === 200) {
                const chatForState: RecentChat = {
                    title: title,
                    conversationID: conversationID,
                    lastModified: currentTime
                };
                setRecentChats(prevState => sortChats([chatForState, ...prevState]))
            }

        } catch (error) {
            console.error("Error creating chat:", error);
            setIsLoading(false)
            return;
        }
    }

    useEffect(() => {
        if (messagesEndRef.current) {
// @ts-ignore
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    const handleModelChange = (event: SelectChangeEvent<string>) => {
        setSelectedModel(event.target.value as string);
        console.log(`Chat model set to: ${AVAILABLE_MODELS.find(m => m.id === event.target.value)?.name}`);
    };

    const handleCopy = async (text: string) => {
        try {
            await navigator.clipboard.writeText(text);
            console.log("Message copied to clipboard!");
        } catch (err) {
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-9999px';
            textArea.style.top = '-9999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand('copy');
                console.log("Message copied via fallback!");
            } catch (fallbackErr) {
                console.error('Fallback copy failed: ', fallbackErr);
            }
            document.body.removeChild(textArea);
        }
    };

    async function sendMessage(){
        if (!input.trim()) return;

        setIsLoading(true);

        if (!sessionStorage.getItem("conversationID")) {
            await createChat(input.trim());
        }


        const userMessage = {
            content: input.trim(),
            sender: 'user',
            message_id: uuidv4(),
            user_id: localStorage.getItem("userID"),
            conversation_id : sessionStorage.getItem("conversationID")
        };

// @ts-ignore
        setMessages(prevMessages => [...prevMessages, userMessage]);

        try {
            const response = await fetch('http://localhost:8000/message/send_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userMessage),
            });

            if (response.status === 200) {
                const llmReply = await response.json()
                const llmResponse = {
                    content : llmReply.content,
                    sender : llmReply.role,
                    message_id : llmReply.message_id
                }

// @ts-ignore
                setMessages(prevMessages => [...prevMessages, llmResponse]);
                setIsLoading(false);
                setInput('');
            }

        } catch (error) {
            console.error("Error sending message:", error);
            setIsLoading(false)
            return;
        }
    }

    const handleNewChat = () => {
        setIsSidebarOpen(false)
        setMessages([]);
        sessionStorage.removeItem("conversationID");
    };

    const handleGoBack = () => {
        if (isLoading) return;
        navigate("/Choice");
    };

    const handleDrawerToggle = () => {
        if (isLoading) return;
        setIsSidebarOpen(!isSidebarOpen);
    };

    const handleSettingsClick = () => {
        if (isLoading) return;
        sessionStorage.setItem("lastPage", "chatbot");
        navigate("/Settings");
    };
    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                height: '100vh',
                width: '100vw',
                background: 'linear-gradient(135deg, #1A2027 0%, #171A21 100%)',
                color: '#e0e0e0',
                fontFamily: 'Roboto, sans-serif',
            }}
        >
            <Box
                onClick={handleDrawerToggle}
                sx={{
                    position: 'fixed',
                    inset: 0,
                    bgcolor: 'rgba(0, 0, 0, 0.5)',
                    zIndex: 99,
                    opacity: isSidebarOpen ? 1 : 0,
                    visibility: isSidebarOpen ? 'visible' : 'hidden',
                    transition: 'opacity 0.3s ease-in-out',
                }}
            />

            <Box
                sx={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    height: '100vh',
                    width: { xs: '70%', sm: 300 },
                    bgcolor: 'rgba(41, 43, 46, 0.95)',
                    transform: isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)',
                    transition: 'transform 0.4s cubic-bezier(0.4, 0.0, 0.2, 1)',
                    zIndex: 100,
                    boxShadow: '4px 0 10px rgba(0, 0, 0, 0.5)',
                    p: 2,
                    display: 'flex',
                    flexDirection: 'column',
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                    }}
                >
                    <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#fff' }}>
                        Conversations
                    </Typography>
                    <IconButton
                        onClick={handleDrawerToggle}
                        disableFocusRipple
                        sx={{
                            color: '#8e8e8e',
                            '&:hover': {
                                color: '#e0e0e0',
                            },
                            '&:focus': { outline: 'none' },
                            '&.Mui-focusVisible': { outline: 'none' }
                        }}
                    >
                        <CloseIcon />
                    </IconButton>
                </Box>

                <FormControl fullWidth variant="filled" sx={{ mb: 1, mt: 2, '.MuiInputLabel-root': { color: '#e0e0e0' }, '.MuiOutlinedInput-notchedOutline': { borderColor: '#5e5e5e' }, '.MuiSelect-select': { color: '#e0e0e0', bgcolor: 'transparent', borderRadius: '8px' } }}>
                    <InputLabel id="model-select-label" sx={{ color: '#fff !important' }}>Select Model</InputLabel>
                    <Select
                        labelId="model-select-label"
                        value={selectedModel}
                        onChange={handleModelChange}
                        label="Select Model"
                        disabled={isLoading}
                        sx={{
                            color: '#e0e0e0',
                            bgcolor: '#3e4042',
                            '.MuiOutlinedInput-notchedOutline': { borderColor: '#5e5e5e' },
                            '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#e0e0e0 !important' },
                            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#1a73e8 !important' },
                            '.MuiSvgIcon-root': { color: '#e0e0e0' }
                        }}
                        MenuProps={{
                            PaperProps: {
                                sx: {
                                    bgcolor: '#3e4042', 
                                    borderRadius: 2,
                                    mt: 0.5,
                                    border: '1px solid #5e5e5e',
                                    color: '#e0e0e0', 
                                },
                            },
                        }}
                    >
                        {AVAILABLE_MODELS.map((model) => (
                           
                            <MenuItem 
                                key={model.id} 
                                value={model.id}
                                sx={{
                                    color: '#e0e0e0',
                                    '&:hover': {
                                        bgcolor: '#5e5e5e',
                                    },
                                    '&.Mui-selected': {
                                        bgcolor: '#1a73e8',
                                        color: '#fff',
                                        '&:hover': {
                                            bgcolor: '#1565c0',
                                        }
                                    }
                                }}
                            >
                                {model.name}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

               
                <Box
                    sx={{
                        flexGrow: 1,
                        overflowY: 'auto',
                        '::-webkit-scrollbar': {
                            display: 'none',
                        },
                        msOverflowStyle: 'none',
                        scrollbarWidth: 'none',
                    }}
                >
                    {recentChats.map(function(chat) {
                        return (
                            <Box
                                key={chat.conversationID}
                                onClick={function() {
                                    if (!isLoading) handleChatClick(chat.conversationID);
                                }}
                                onContextMenu={function(e) {
                                    handleRightClick(e, chat.conversationID);
                                }}
                                sx={{
                                    p: 1.5,
                                    my: 1,
                                    bgcolor: '#3e4042',
                                    borderRadius: 2,
                                    opacity: isLoading ? 0.5 : 1,
                                    '&:hover': {
                                        bgcolor: isLoading ? '#3e4042' : '#5e5e5e',
                                        cursor: isLoading ? 'default' : 'pointer',
                                    },
                                }}
                            >
                                <Typography variant="body1">
                                    {chat.title}
                                </Typography>
                            </Box>
                        );
                    })}
                </Box>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, p: 1, marginTop: 'auto', marginBottom: 2 }}>
                    <Button
                        onClick={handleNewChat}
                        disabled={isLoading}
                        variant="text"
                        startIcon={<CreateIcon />}
                        sx={{
                            color: '#e0e0e0',
                            justifyContent: 'flex-start',
                            textTransform: 'none',
                            py: 1.5,
                            '&:hover': {
                                bgcolor: '#3e4042',
                            },
                            '&:focus': { outline: 'none' },
                            '&.Mui-focusVisible': { backgroundColor: 'transparent' },
                        }}
                    >
                        New Conversation
                    </Button>
                    <Button
                        onClick={handleSettingsClick}
                        disabled={isLoading}
                        variant="text"
                        startIcon={<SettingsIcon />}
                        sx={{
                            color: '#e0e0e0',
                            justifyContent: 'flex-start',
                            textTransform: 'none',
                            py: 1.5,
                            '&:hover': {
                                bgcolor: '#3e4042',
                            },
                            '&:focus': { outline: 'none' },
                            '&.Mui-focusVisible': { backgroundColor: 'transparent' },
                        }}
                    >
                        Settings
                    </Button>
                    <Button
                        onClick={handleGoBack}
                        disabled={isLoading}
                        variant="text"
                        startIcon={<ExitToAppIcon />}
                        sx={{
                            color: '#e0e0e0',
                            justifyContent: 'flex-start',
                            textTransform: 'none',
                            py: 1.5,
                            '&:hover': {
                                bgcolor: '#3e4042',
                            },
                            '&:focus': { outline: 'none' },
                            '&.Mui-focusVisible': { backgroundColor: 'transparent' },
                        }}
                    >
                        Exit
                    </Button>
                </Box>
            </Box>

            <Box
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    flexGrow: 1,
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        p: 2,
                        bgcolor: 'rgba(41, 43, 46, 0.8)',
                        backdropFilter: 'blur(10px)',
                        borderBottom: '1px solid #3e4042',
                        boxShadow: '0px 2px 10px rgba(0, 0, 0, 0.5)',
                        position: 'sticky',
                        top: 0,
                        zIndex: 10,
                    }}
                >
                    <IconButton
                        onClick={handleDrawerToggle}
                        disabled={isLoading}
                        disableFocusRipple
                        sx={{
                            color: '#8e8e8e',
                            '&:hover': {
                                color: isLoading ? '#8e8e8e' : '#e0e0e0',
                            },
                            '&:focus': { outline: 'none' },
                            '&.Mui-focusVisible': { outline: 'none' },
                            opacity: isLoading ? 0.5 : 1,
                            cursor: isLoading ? 'default' : 'pointer',
                        }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Box
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            position: 'absolute',
                            left: '50%',
                            transform: 'translateX(-50%)',
                        }}
                    >
                        <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 'bold' }}>
                            Athena
                        </Typography>
                        <Box
                            sx={{
                                width: 16,
                                height: 16,
                                borderRadius: '50%',
                                bgcolor: '#1a73e8',
                                ml: 1,
                            }}
                        />
                    </Box>
                </Box>

                <Box
                    sx={{
                        flexGrow: 1,
                        overflowY: 'auto',
                        p: 3,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                        scrollBehavior: 'smooth',
                        '::-webkit-scrollbar': {
                            display: 'none',
                        },
                        msOverflowStyle: 'none',
                        scrollbarWidth: 'none',
                    }}
                >
                    {messages.length === 0 ? (
                        <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <Typography variant="h4" sx={{ color: '#5e5e5e' }}>
                                How can I help you today {localStorage.getItem("firstName")}?
                            </Typography>
                        </Box>
                    ) : (
                        messages.map((msg, index) => (
                            <Box
                                key={msg.id}
                                sx={{
                                    display: 'flex',
                                    justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                                }}
                            >
                                <Box
                                    sx={{
                                        display: 'flex',
                                        flexDirection: 'column',
                                        maxWidth: '70%',
                                    }}
                                >
                                    <Box
                                        sx={{
                                            p: 2,
                                            borderRadius: msg.sender === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                                            bgcolor: msg.sender === 'user' ? '#1a73e8' : '#3e4042',
                                            color: '#e0e0e0',
                                            boxShadow: '0px 2px 5px rgba(0, 0, 0, 0.3)',
                                        }}
                                    >
                                        <Typography variant="body1">{msg.content}</Typography>
                                    </Box>

                                    
                                    {index >= messages.length - 1 && (
                                        <IconButton
                                            size="small"
                                            onClick={() => handleCopy(msg.content)}
                                            sx={{
                                                alignSelf: 'flex-end',
                                                mt: 0.5,
                                                color: '#8e8e8e',
                                                p: 0.5,
                                                '&:hover': { color: '#e0e0e0' },
                                            }}
                                        >
                                            <ContentCopyIcon fontSize="small" />
                                        </IconButton>
                                    )}
                                </Box>
                            </Box>
                        ))
                    )}
                    {isLoading && (
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'flex-start',
                                p: 2,
                                alignItems: 'center',
                                gap: 2,
                            }}
                        >
                            <Box
                                sx={{
                                    width: '20px',
                                    height: '20px',
                                    border: '3px solid #5e5e5e',
                                    borderTop: '3px solid #1a73e8',
                                    borderRadius: '50%',
                                    animation: 'spin 1s linear infinite',
                                    '@keyframes spin': {
                                        '0%': { transform: 'rotate(0deg)' },
                                        '100%': { transform: 'rotate(360deg)' },
                                    },
                                }}
                            />
                            <Typography
                                variant="body1"
                                sx={{ color: '#8e8e8e', fontStyle: 'italic' }}
                            >
                                {currentLoadingMessage}
                            </Typography>
                        </Box>
                    )}
                    <div ref={messagesEndRef} />
                </Box>

                <Box
                    sx={{
                        p: 2,
                        bgcolor: 'rgba(41, 43, 46, 0.8)',
                        backdropFilter: 'blur(10px)',
                        borderTop: '1px solid #3e4042',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2,
                        boxShadow: '0px -2px 10px rgba(0, 0, 0, 0.5)',
                    }}
                >
                    <TextField
                        fullWidth
                        multiline
                        maxRows={4}
                        variant="outlined"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                sendMessage();
                            }
                        }}
                        placeholder="Enter your question here"
                        disabled={isLoading}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                borderRadius: 4,
                                bgcolor: '#3e4042',
                                '& fieldset': {
                                    borderColor: 'transparent',
                                },
                                '&:hover fieldset': {
                                    borderColor: '#5e5e5e',
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: '#1a73e8',
                                },
                                '&.Mui-disabled': {
                                    bgcolor: '#2e2e2e',
                                    opacity: 0.6,
                                },
                            },
                            '& .MuiInputBase-input': {
                                color: '#e0e0e0',
                            },
                        }}
                    />
                    <IconButton
                        color="primary"
                        onClick={sendMessage}
                        disabled={!input.trim() || isLoading}
                        sx={{
                            p: 1.5,
                            bgcolor: '#1a73e8',
                            '&:hover': {
                                bgcolor: '#1565c0',
                            },
                            '&.Mui-disabled': {
                                bgcolor: '#5e5e5e',
                                color: '#a0a0a0',
                            },
                        }}
                    >
                        <SendIcon sx={{ color: '#fff' }} />
                    </IconButton>
                </Box>
            </Box>
        </Box>
    );
};

export default ChatBot;