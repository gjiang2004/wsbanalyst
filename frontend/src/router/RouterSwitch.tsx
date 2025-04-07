import { Route, Routes } from "react-router-dom";
import { pageHome, pageTopPosts, pageWsbChatbot, pageTrading } from "./router"
import { Home } from "../components/home/home"
import { TopPosts } from "../components/topposts/topposts"
import { WsbChatbot } from "../components/wsbchatbot/wsbchatbot"
import { Trading } from "../components/trading/trading"

export default function RouterSwitch() {
    return (
        <Routes>
            <Route path={pageHome} element={<Home />} />
            <Route path={pageWsbChatbot} element={<WsbChatbot />} />
            <Route path={pageTopPosts} element={<TopPosts />} />
            <Route path={pageTrading} element={<Trading />} />
        </Routes>
    )
}