import { BrowserRouter, Routes, Route } from "react-router-dom";
import Registration from "./Registration.tsx";
import Login from "./Login.tsx";
import ChatBot from "./ChatBot.tsx";
import Loading from "./Loading.tsx";
import Choice from "./Choice.tsx";
import StoreKey from "./StoreKey.tsx";
import PDFManager from "./PDFManager.tsx";
import Settings from "./Settings.tsx";
const App = () => (
    <div className="App">
        <BrowserRouter>
            <Routes>
                <Route index element={<Loading/>} />
                <Route path="chatbot" element={<ChatBot/>} />
                <Route path="login" element={<Login/>} />
                <Route path="registration" element={<Registration/>} />
                <Route path="choice" element={<Choice/>} />
                <Route path="storeKey" element={<StoreKey/>} />
                <Route path="pdfManager" element={<PDFManager/>} />
                <Route path="settings" element={<Settings/>} />
            </Routes>
        </BrowserRouter>
    </div>
);

export default App;
