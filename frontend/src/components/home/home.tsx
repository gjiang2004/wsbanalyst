import { useState, useRef } from 'react'
import wsbLogo from '../../assets/WSB.png'
import downArrow from '../../assets/down.png'
import stock from '../../assets/stock.png'
//import community from '../../assets/community.png'
import chatbot from '../../assets/chatbot.png'
import { useNavigate } from 'react-router-dom';
import { pageTopPosts , pageWsbChatbot, pageTrading } from '../../router/router';
import './home.css'

export const Home = () => {
  const [count, setCount] = useState(0)
  const featuresRef = useRef(null)
  
  const scrollToFeatures = () => {
    featuresRef.current?.scrollIntoView({ behavior: 'smooth' });
  }
  const navigate = useNavigate()

  const handleClick = (url: string) => {
      navigate(url)
  }
  
  return (
    <>
      <div className="top">
        <a href="https://www.reddit.com/r/wallstreetbets/" target="_blank" rel="noopener noreferrer">
          <img src={wsbLogo} className="logo" alt="WSB logo" style={{ width: '20rem', height: 'auto' }} /> 
        </a>
        <h1>HackPSU 2025 r/wallstreetbets Analyst</h1>
        <img 
          src={downArrow} 
          className="down" 
          alt="down arrow" 
          onClick={scrollToFeatures}
        />
      </div>
      <div>
      <div className="content-section" ref={featuresRef}>
        <h2>Market Analysis Tools</h2>
        <p>Advanced tools powered by r/wallstreetbets sentiment analysis</p>
        <br></br>
        <br></br>
        <div className="features">
          {/*
          <div className="feature-card" onClick={() => handleClick(pageTopPosts)}>
            <div className="feature-icon"><img src={community} alt="community logo" style={{ width: '12rem', height: 'auto' }} /></div>
            <h3>Community Insights</h3>
            <p>Stay updated with the latest WSB discussions and trends</p>
          </div>
          */}
          <div className="feature-card" onClick={() => handleClick(pageWsbChatbot)}>
            <div className="feature-icon"><img src={chatbot} alt="chatbot logo" style={{ width: '12rem', height: 'auto' }} /></div>
            <h3>AI Chatbot</h3>
            <p>Interact with our WSB-trained AI assistant for investment ideas</p>
          </div>

          <div className="feature-card" onClick={() => handleClick(pageTrading)}>
            <div className="feature-icon"><img src={stock} alt="stock logo" style={{ width: '12rem', height: 'auto' }} /></div>
            <h3>Trade Analysis</h3>
            <p>See a stock trading simulation created using WSB posts</p>
          </div>
        </div>
        
      </div>
      <div>
        <p className="disclaimer">
          Disclaimer: This is the best financial advice you can find. Never do your own research.
        </p>
      </div>
      <footer>
        <p>HackPSU 2025 | r/wallstreetbets Analyst</p>
      </footer>
      </div>
    </>
  )
};

export default Home