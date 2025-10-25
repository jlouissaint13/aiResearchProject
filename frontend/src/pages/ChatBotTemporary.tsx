import { useState, useEffect, useRef } from 'react';
import { Box, Button, Typography, TextField, IconButton, Divider } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import SettingsIcon from '@mui/icons-material/Settings';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import AssistantIcon from '@mui/icons-material/Assistant';

const LOADING_MESSAGES = [
    "Consulting the data...",
    "Synthesizing your request...",
    "Formulating a response...",
    "Almost there! Just a moment...",
    "Finalizing everything for you..."
];

// Removed AVAILABLE_MODELS constant

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
    // Removed selectedModel state
    const [isLoggedIn,setIsLoggedIn] = useState<boolean>()

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

    useEffect(() => {
        loginStatus()
    }, []);
    function loginStatus() : void {
        if (localStorage.getItem("loggedIn") === "true") {
            setIsLoggedIn(true);
            return
        }

        setIsLoggedIn(false)

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
            user_id : localStorage.getItem("none")
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
            alert("Error loading conversations")
            return;
        }
    }

    async function handleChatClick(conversationID:string) {
        setIsSidebarOpen(false);
        setMessages([]);

        sessionStorage.setItem("conversationID", conversationID);

        const data = {
            conversation_id: conversationID,
            user_id: localStorage.getItem("none")
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
                user_id : localStorage.getItem("none")
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
        handleDeleteChat(conversationID);
    }

    async function createChat(title : string) {
        let conversationID = uuidv4();
        sessionStorage.setItem("conversationID",conversationID);
        sessionStorage.setItem("mostRecentID",conversationID)
        const currentTime = new Date().toISOString();
        const newChatDataForBackend = {
            user_id: localStorage.getItem("none"),
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

    // Removed handleModelChange handler

    const handleCopy = async (text: string) => {
        try {
            await navigator.clipboard.writeText(text);
            alert("Copied to clipboard!");
        } catch (err) {

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
            user_id: localStorage.getItem("none"),
            conversation_id : sessionStorage.getItem("conversationID"),
            logged_in : localStorage.getItem("loggedIn")
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
                height: '100vh',
                width: '100vw',
                overflow: 'hidden',
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
                    zIndex: 1100,
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
                    width: { xs: '75%', sm: 320 },
                    bgcolor: 'rgba(30, 32, 35, 0.98)',
                    backdropFilter: 'blur(8px)',
                    transform: isSidebarOpen ? 'translateX(0)' : 'translateX(-100%)',
                    transition: 'transform 0.4s cubic-bezier(0.4, 0.0, 0.2, 1)',
                    zIndex: 1200,
                    boxShadow: '4px 0 15px rgba(0, 0, 0, 0.5)',
                    borderRight: '1px solid rgba(255, 255, 255, 0.05)',
                    p: 2,
                    display: 'flex',
                    flexDirection: 'column',
                }}
            >
                <Box sx={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 2, mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="h6" sx={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5, color: '#e0e0e0' }}>
                            Conversations
                        </Typography>
                        <IconButton
                            onClick={handleDrawerToggle}
                            sx={{
                                color: '#8e8e8e',
                                '&:hover': { color: '#e0e0e0', bgcolor: 'transparent' },
                                '&:active': { bgcolor: 'transparent' },
                                outline: 'none',
                                '&:focus, &.Mui-focusVisible': { bgcolor: 'transparent', boxShadow: 'none', outline: 'none' }
                            }}
                            disableRipple
                            disableFocusRipple
                            disableTouchRipple
                        >
                            <CloseIcon />
                        </IconButton>
                    </Box>

                    {/* Removed Model Selector FormControl */}
                </Box>

                <Box sx={{ flexGrow: 1, overflowY: 'auto', pr: 1, mr: -1, minHeight: 0 }}>
                    {recentChats.map((chat) => (
                        <Button
                            key={chat.conversationID}
                            fullWidth
                            onClick={() => { if (!isLoading) handleChatClick(chat.conversationID); }}
                            onContextMenu={(e) => handleRightClick(e, chat.conversationID)}
                            disableRipple
                            disableFocusRipple
                            disableTouchRipple
                            sx={{
                                justifyContent: 'flex-start',
                                p: 1.5,
                                my: 0.5,
                                borderRadius: 1,
                                bgcolor: 'transparent',
                                opacity: isLoading ? 0.6 : 1,
                                transition: 'background-color 0.3s, box-shadow 0.1s, transform 0.1s',
                                textTransform: 'none',
                                outline: 'none',
                                '&:focus, &.Mui-focusVisible': { bgcolor: 'transparent', boxShadow: 'none', outline: 'none' },
                                '&:hover': {
                                    bgcolor: isLoading ? 'transparent' : '#282a2e',
                                    cursor: isLoading ? 'default' : 'pointer',
                                    transform: isLoading ? 'none' : 'translateY(-1px)',
                                    boxShadow: isLoading ? 'none' : '0 2px 5px rgba(0, 0, 0, 0.3)',
                                },
                            }}
                        >
                            <Typography noWrap variant="body1" sx={{ color: '#e0e0e0', fontWeight: 500 }}>
                                {chat.title}
                            </Typography>
                        </Button>
                    ))}
                    {recentChats.length === 0 && (
                        <Typography sx={{ color: '#8e8e8e', textAlign: 'center', mt: 3, fontStyle: 'italic' }}>
                            Conversation history is disabled while temporary mode is active.
                        </Typography>
                    )}
                </Box>

                <Box sx={{ flexShrink: 0, pt: 1, mt: 1, pb: 2.9 }}>
                    <Divider sx={{ mb: 1, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        {[
                            { text: 'Settings', icon: <SettingsIcon />, handler: handleSettingsClick, visible: isLoggedIn },
                            { text: 'Exit', icon: <ExitToAppIcon />, handler: handleGoBack, visible: true }
                        ].map((item) => item.visible && (
                            <Button
                                key={item.text}
                                onClick={item.handler}
                                disabled={isLoading}
                                variant="text"
                                startIcon={item.icon}
                                disableRipple
                                disableFocusRipple
                                disableTouchRipple
                                sx={{
                                    color: '#e0e0e0',
                                    justifyContent: 'flex-start',
                                    textTransform: 'none',
                                    py: 1.25,
                                    px: 1.5,
                                    borderRadius: 1,
                                    fontWeight: 500,
                                    bgcolor: 'transparent',
                                    '& .MuiSvgIcon-root': { color: '#1a73e8' },
                                    transition: 'background-color 0.3s, box-shadow 0.1s, transform 0.1s',
                                    outline: 'none',
                                    '&:focus, &.Mui-focusVisible': { bgcolor: 'transparent', boxShadow: 'none', outline: 'none' },
                                    '&:hover': {
                                        bgcolor: '#282a2e',
                                        transform: 'translateY(-1px)',
                                        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)',
                                    },
                                    '&:active': { bgcolor: 'transparent' }
                                }}
                            >
                                {item.text}
                            </Button>
                        ))}
                    </Box>
                </Box>
            </Box>

            <Box sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, height: '100vh', transition: 'margin-left 0.4s cubic-bezier(0.4, 0.0, 0.2, 1)' }}>
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        p: 2,
                        bgcolor: 'rgba(30, 32, 35, 0.98)',
                        borderBottom: '1px solid #3e4042',
                        boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
                        flexShrink: 0,
                    }}
                >
                    <IconButton
                        onClick={handleDrawerToggle}
                        disabled={isLoading}
                        sx={{
                            color: '#8e8e8e',
                            '&:hover': { color: '#e0e0e0', bgcolor: 'transparent' },
                            '&:active': { bgcolor: 'transparent' },
                            outline: 'none',
                            '&:focus, &.Mui-focusVisible': {
                                bgcolor: 'transparent',
                                boxShadow: 'none',
                                outline: 'none'
                            }
                        }}
                        disableRipple
                        disableFocusRipple
                        disableTouchRipple
                    >
                        <MenuIcon />
                    </IconButton>
                    <Box sx={{ display: 'flex', alignItems: 'center', position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
                        <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            Marie
                        </Typography>
                        <AssistantIcon sx={{ color: '#1a73e8', ml: 1.5 }} />
                    </Box>
                </Box>

                <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {messages.length === 0 ? (
                        <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                            <Typography variant="h5" sx={{ color: '#8e8e8e' }}>
                                How can I help you today{localStorage.getItem("firstName") ? `, ${localStorage.getItem("firstName")}` : ''}?
                            </Typography>
                        </Box>
                    ) : (
                        messages.map((msg, index) => (
                            <Box key={msg.id} sx={{ display: 'flex', justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                                <Box sx={{ display: 'flex', flexDirection: 'column', maxWidth: '80%', alignItems: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                                    <Box
                                        sx={{
                                            p: 1.5,
                                            borderRadius: msg.sender === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                                            bgcolor: msg.sender === 'user' ? '#1a73e8' : '#282a2e',
                                            color: msg.sender === 'user' ? '#fff' : '#e0e0e0',
                                            boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.2)',
                                        }}
                                    >
                                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', fontWeight: 500 }}>{msg.content}</Typography>
                                    </Box>

                                    {index === messages.length - 1 && (
                                        <IconButton
                                            size="small"
                                            onClick={() => handleCopy(msg.content)}
                                            disableRipple
                                            disableFocusRipple
                                            disableTouchRipple
                                            sx={{
                                                mt: 0.5,
                                                alignSelf: 'flex-end',
                                                color: '#8e8e8e',
                                                '&:hover': { color: '#e0e0e0', bgcolor: 'transparent' },
                                                '&:active': { bgcolor: 'transparent' },
                                                outline: 'none',
                                                '&:focus, &.Mui-focusVisible': { bgcolor: 'transparent', boxShadow: 'none', outline: 'none' }
                                            }}
                                        >
                                            <ContentCopyIcon fontSize="inherit" />
                                        </IconButton>
                                    )}
                                </Box>
                            </Box>
                        ))
                    )}
                    {isLoading && (
                        <Box sx={{ display: 'flex', justifyContent: 'flex-start', p: 2, alignItems: 'center', gap: 2 }}>
                            <Box sx={{ width: '20px', height: '20px', border: '3px solid #3e4042', borderTop: '3px solid #1a73e8', borderRadius: '50%', animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
                            <Typography variant="body1" sx={{ color: '#8e8e8e', fontStyle: 'italic' }}>
                                {currentLoadingMessage}
                            </Typography>
                        </Box>
                    )}
                    <div ref={messagesEndRef} />
                </Box>

                <Box sx={{ p: 2, bgcolor: 'rgba(30, 32, 35, 0.98)', borderTop: '1px solid #3e4042', display: 'flex', alignItems: 'center', gap: 2, flexShrink: 0, boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.3)' }}>
                    <TextField
                        fullWidth
                        multiline
                        maxRows={5}
                        variant="outlined"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                        placeholder="Enter your question here"
                        disabled={isLoading}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                borderRadius: 4,
                                bgcolor: '#282a2e',
                                '& fieldset': { borderColor: '#3e4042' },
                                '&:hover fieldset': { borderColor: '#5e6062' },
                                '&.Mui-focused fieldset': { borderColor: '#1a73e8', borderWidth: '2px' },
                            },
                            '& .MuiInputBase-input': { color: '#e0e0e0', '::-webkit-scrollbar': { display: 'none' }, msOverflowStyle: 'none', scrollbarWidth: 'none' },
                        }}
                    />
                    <IconButton
                        color="primary"
                        onClick={sendMessage}
                        disabled={!input.trim() || isLoading}
                        disableRipple
                        disableFocusRipple
                        disableTouchRipple
                        sx={{
                            p: 1.5,
                            bgcolor: '#1a73e8',
                            color: '#fff',
                            transition: 'background-color 0.3s, box-shadow 0.1s, transform 0.1s',
                            outline: 'none',
                            '&:focus, &.Mui-focusVisible': { bgcolor: '#1a73e8', boxShadow: 'none', outline: 'none' },
                            '&:hover': {
                                bgcolor: '#1565c0',
                                transform: 'translateY(-1px)',
                                boxShadow: '0 4px 8px rgba(0, 0, 0, 0.4)',
                            },
                            '&.Mui-disabled': { bgcolor: '#282a2e', color: '#8e8e8e' },
                            '&:active': { bgcolor: '#1a73e8' }
                        }}
                    >
                        <SendIcon />
                    </IconButton>
                </Box>
            </Box>
        </Box>
    );
};

export default ChatBot;