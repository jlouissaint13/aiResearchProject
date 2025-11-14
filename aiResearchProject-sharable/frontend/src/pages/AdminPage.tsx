import { useEffect, useState } from "react";
import {Box, Typography, Button, Table, TableBody, TableCell, TableHead, TableRow, Switch, Divider
} from "@mui/material";

import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from "react-router-dom";

export default function AdminPage() {
    
    const [guestEnabled, setGuestEnabled] = useState(true);
    const [message, setMessage] = useState("");

   interface User {
     user_id: string;
     username: string;
     email: string;
     role: string;
     is_active: boolean;
   }

   const [users, setUsers] = useState<User[]>([]);

   interface PendingAdmin {
    request_id: number;
    first_name: string;
    email: string;
    username: string;
   }

   const [pendingAdmins, setPendingAdmins] = useState<PendingAdmin[]>([]);

   
    const navigate = useNavigate();

    // Replace with the username of the logged-in admin; later we’ll hook this up to your login state.
    const acting = "Dude1";

    useEffect(() => {
        fetch(`http://localhost:8000/admin/users?acting=${acting}`)
            .then(res => res.json())
            .then(data => setUsers(data))
            .catch(() => setMessage("Failed to load users"));

        fetch("http://localhost:8000/admin/flags/guest_enabled")
            .then(res => res.json())
            .then(data => setGuestEnabled(data.guest_enabled))
            .catch(() => setMessage("Failed to load guest flag"));

        fetch(`http://localhost:8000/admin/pending_admins?acting=${acting}`)
            .then(res => res.json())
            .then(data => {
               console.log("Pending admin data:", JSON.stringify(data, null, 2));
               setPendingAdmins(data);
           })
           .catch(() => setMessage("Failed to load pending admins"));

    }, []);

    const toggleGuest = async () => {
        const newValue = !guestEnabled;
        const res = await fetch("http://localhost:8000/admin/flags/guest_enabled", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ acting, enable: newValue }),
        });
        if (res.ok) setGuestEnabled(newValue);
    };

    const deactivateUser = async (userId: string, enable: boolean) => {
        const res = await fetch(`http://localhost:8000/admin/users/${userId}/status`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ acting, enable }),
        });
        if (res.ok) {
            setUsers(users.map(u => u.user_id === userId ? { ...u, is_active: enable } : u));
        }
    };

    const deleteUser = async (userId: string) => {
        const confirmDelete = window.confirm(
            "Are you sure you want to permanently delete this user account? This action cannot be undone."
         );
         if (!confirmDelete) return;    
        const res = await fetch(`http://localhost:8000/admin/users/${userId}`, {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ acting }),
        });
        if (res.ok) {
            setUsers(users.filter((u) => u.user_id !== userId));
        }
    };

    const approvePendingAdmin = async (requestId: string) => {
      const res = await fetch(`http://localhost:8000/admin/pending_admins/${requestId}/approve`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ acting }),
      });
      if (res.ok) {

         setPendingAdmins(pendingAdmins.filter((p) => p.request_id !== Number(requestId)));


         fetch(`http://localhost:8000/admin/users?acting=${acting}`)
          .then((res) => res.json())
          .then((data) => setUsers(data))
           .catch(() => setMessage("Failed to refresh users after approval"));

         alert("Admin request approved!");
        } else {
          alert("Failed to approve admin request.");
        }
      };


    

    const buttonStyle = {
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
    };

    return (
        <Box
            sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "flex-start",
                position: "absolute",
                inset: 0,
                background: "linear-gradient(135deg, #1A2027 0%, #171A21 100%)",
                color: "#e0e0e0",
                fontFamily: "Roboto, sans-serif",
                p: 3,
                overflowY: "auto", 
            }}
        >
            {/* Back to Dashboard Button */}
            <Button
                variant="contained"
                onClick={() => navigate("/choice")}
                startIcon={<ArrowBackIcon />}
                sx={{
                    position: 'absolute',
                    top: 20,
                    left: 20,
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
                Back to Dashboard
            </Button>

            <Box
                sx={{
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
                    maxWidth: 700,
                    border: "1px solid rgba(255, 255, 255, 0.05)",
                }}
            >
                <Typography
                    variant="h5"
                    component="h1"
                    sx={{
                        color: "#e0e0e0",
                        fontWeight: 600,
                        letterSpacing: 0.5,
                        textTransform: "uppercase",
                        mb: 2,
                    }}
                >
                    Admin Dashboard
                </Typography>

                {/* Guest Switch */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Typography variant="body1" sx={{ color: "#8e8e8e" }}>
                        Guest Features:
                    </Typography>
                    <Switch
                        checked={guestEnabled}
                        onChange={toggleGuest}
                        sx={{
                            "& .MuiSwitch-thumb": { bgcolor: "#1a73e8" },
                            "& .MuiSwitch-track": { bgcolor: "#3e4042" },
                        }}
                    />
                </Box>

                {/* API Key Management Section */}
                <Divider sx={{ width: "100%", bgcolor: "rgba(255, 255, 255, 0.08)", my: 2 }} />
                <Typography
                    variant="h6"
                    sx={{
                        color: "#1a73e8",
                        fontWeight: 600,
                        textTransform: "uppercase",
                        letterSpacing: 0.5,
                        mb: 1,
                    }}
                >
                    API Keys
                </Typography>

                <Box
                    sx={{
                        width: "100%",
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                        p: 3,
                        border: "1px solid #3e4042",
                        borderRadius: 1,
                        bgcolor: "#282a2e",
                    }}
                >
                    

                    <Button
                        fullWidth
                        variant="contained"
                        sx={buttonStyle}
                        onClick={() => navigate("/StoreKey")}
                    >
                        Manage API Keys
                    </Button>
                </Box>

                {/* User Management Table */}
                <Divider sx={{ width: "100%", bgcolor: "rgba(255, 255, 255, 0.08)", my: 2 }} />
                <Typography
                    variant="h6"
                    sx={{
                        color: "#1a73e8",
                        fontWeight: 600,
                        textTransform: "uppercase",
                        letterSpacing: 0.5,
                        mb: 1,
                    }}
                >
                    User Management
                </Typography>

                <Table
                    sx={{
                        width: "100%",
                        backgroundColor: "transparent",
                        "& th": { color: "#8e8e8e" },
                        "& td": { color: "#e0e0e0" },
                        borderCollapse: "collapse",
                    }}
                >
                    <TableHead>
                        <TableRow>
                            <TableCell>Username</TableCell>
                            <TableCell>Email</TableCell>
                            <TableCell>Role</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell align="center">Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {users.map((u) => (
                            <TableRow key={u.user_id}>
                                <TableCell>{u.username}</TableCell>
                                <TableCell>{u.email}</TableCell>
                                <TableCell>{u.role}</TableCell>
                                <TableCell>{u.is_active ? "Active" : "Disabled"}</TableCell>
                                <TableCell align="center">
                                    <Button
                                        variant="contained"
                                        onClick={() => deactivateUser(u.user_id, !u.is_active)}
                                        sx={{
                                            p: 0.75,
                                            borderRadius: 1,
                                            bgcolor: u.is_active ? "#2f25beff": "#1a73e8",
                                            color: "#fff",
                                            textTransform: "none",
                                            fontWeight: 500,
                                            transition: "background-color 0.3s, box-shadow 0.1s",
                                            "&:hover": {
                                                bgcolor: u.is_active ? "#2f25beff" : "#1565c0",
                                                transform: "translateY(-1px)",
                                                boxShadow: "0 2px 5px rgba(0, 0, 0, 0.3)",
                                            },
                                        }}
                                    >
                                        {u.is_active ? "Disable" : "Enable"}
                                    </Button>
                                
                                <Button
                                    variant="contained"
                                    onClick={() => deleteUser(u.user_id)}
                                    sx={{
                                        ml: 1.5,
                                        p: 0.75,
                                        borderRadius: 1,
                                        bgcolor: "#d32f2f",
                                        color: "#fff",
                                        textTransform: "none",
                                        fontWeight: 500,
                                        transition: "background-color 0.3s, box-shadow 0.1s",
                                        "&:hover": {
                                            bgcolor: "#b71c1c",
                                            transform: "translateY(-1px)",
                                            boxShadow: "0 2px 5px rgba(0, 0, 0, 0.3)",
                                       },
                                    }}
                                 >
                                    Delete
                                </Button>
                            </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>

                {/* Pending Admin Approvals */}
                <Divider sx={{ width: "100%", bgcolor: "rgba(255, 255, 255, 0.08)", my: 2 }} />
                <Typography
                  variant="h6"
                  sx={{
                    color: "#1a73e8",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                    mb: 1,
                  }}
                >
                  Pending Admin Approvals
                </Typography>

                <Table
                  sx={{
                    width: "100%",
                    backgroundColor: "transparent",
                    "& th": { color: "#8e8e8e" },
                    "& td": { color: "#e0e0e0" },
                    borderCollapse: "collapse",
                  }}
                >
                  <TableHead>
                    <TableRow>
                      <TableCell>First Name</TableCell>
                      <TableCell>Email</TableCell>
                      <TableCell>Username</TableCell>
                      <TableCell align="center">Action</TableCell>
                     </TableRow>
                     </TableHead>
                    
                     <TableBody>
                        {pendingAdmins.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={4} align="center" sx={{ color: "#8e8e8e" }}>
                              No pending admin requests.
                            </TableCell>
                           </TableRow>
                         ) : (
                           pendingAdmins.map((p) => (
                             <TableRow key={p.request_id}>
                               <TableCell>{p.first_name}</TableCell>
                               <TableCell>{p.email}</TableCell>
                               <TableCell>{p.username}</TableCell>
                              <TableCell align="center">
                                <Button
                                  variant="contained"
                                  sx={{
                                    p: 0.75,
                                    borderRadius: 1,
                                    bgcolor: "#1a73e8",
                                    color: "#fff",
                                    textTransform: "none",
                                    fontWeight: 500,
                                    transition: "background-color 0.3s, box-shadow 0.1s",
                                    "&:hover": {
                                      bgcolor: "#1565c0",
                                      transform: "translateY(-1px)",
                                      boxShadow: "0 2px 5px rgba(0, 0, 0, 0.3)",
                                    },
                                   }}
                                   onClick={() => approvePendingAdmin(String(p.request_id))}

                                 >
                                   Approve
                                 </Button>
                               </TableCell>
                              </TableRow>
                            ))
                          )}
                        </TableBody>
                       </Table>
                    

                {message && (
                    <Typography variant="body2" sx={{ color: "#f44336" }}>
                        {message}
                    </Typography>
                )}
            </Box>
        </Box>
    );
}
