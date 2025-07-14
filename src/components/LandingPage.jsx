import React, { useEffect, useState } from 'react';
import '../styles/LandingPage.css';
import { useNavigate } from 'react-router-dom';
import Header from "./Header";
import Footer from "./Footer";
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

function LandingPage() {
  const navigate = useNavigate();
  const handleLoginClick = () => {
    navigate("/login");
  };

  const images = [
    "/images/bauxite1.jpg",
    "/images/bauxite2.jpg",
    "/images/bauxite3.jpg",
    "/images/bauxite4.jpg",
    "/images/bauxite5.jpg"
  ];

  const [detectionAccuracy, setDetectionAccuracy] = useState(0);
  const [minersMonitored, setMinersMonitored] = useState(0);
  const [showTagline, setShowTagline] = useState(false);

  useEffect(() => {
    let acc = 0;
    let miners = 0;
    const interval = setInterval(() => {
      if (acc < 85) {
        acc += 1;
        setDetectionAccuracy(acc);
      }
      if (miners < 500) {
        miners += 10;
        setMinersMonitored(miners > 500 ? 500 : miners);
      }
      if (acc >= 85 && miners >= 500) {
        clearInterval(interval);
      }
    }, 30);
    return () => clearInterval(interval);
  }, []);

  // Show tagline after bullet animation (6s total)
  useEffect(() => {
    const timeout = setTimeout(() => {
      setShowTagline(true);
    }, 6000);
    return () => clearTimeout(timeout);
  }, []);

  const [currentSlide, setCurrentSlide] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % images.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [images.length]);

  const nextSlide = () => setCurrentSlide((prev) => (prev + 1) % images.length);
  const prevSlide = () => setCurrentSlide((prev) => (prev - 1 + images.length) % images.length);
  const goToSlide = (index) => setCurrentSlide(index);

  return (
    <>
      <Header />

      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div className="hero-text">
            <h2 className="typing-title">RespiGuard AI</h2>
            <p className="fade-in">AI-powered Respiratory Health Monitoring for NALCO Bauxite Mines</p>
            <div className="main-aims">
              <p className="bullet-line">• Early detection of respiratory diseases.</p>
              <p className="bullet-line">• Protect workers in hazardous mining environments.</p>
              <p className="bullet-line">• Provide real-time health analytics to management.</p>
            </div>

            <br/>
            <br/>

            <button className="animated-button" onClick={handleLoginClick}>
                <svg xmlns="http://www.w3.org/2000/svg" className="arr-2" viewBox="0 0 24 24">
                <path d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"></path>
                </svg>
                <span className="text">Login to SCAN</span>
                <span className="circle"></span>
                <svg xmlns="http://www.w3.org/2000/svg" className="arr-1" viewBox="0 0 24 24">
                <path d="M16.1716 10.9999L10.8076 5.63589L12.2218 4.22168L20 11.9999L12.2218 19.778L10.8076 18.3638L16.1716 12.9999H4V10.9999H16.1716Z"></path>
                </svg>
            </button>
            
            {/* Stats Section */}
            <div className="stats-section">
              <div className="stat-item">
                <h2>{detectionAccuracy}%</h2>
                <p><em>Detection Accuracy</em></p>
              </div>
              <div className="stat-item">
                <h2>{minersMonitored}+</h2>
                <p><em>Miners Monitored</em></p>
              </div>
              <div className="stat-item">
                <h2>Live</h2>
                <p><em>Health Alerts</em></p>
              </div>
            </div>
          </div>

          <div className="hero-image">
            <img src="/images/logo.jpg" alt="Lungs Analysis" />
          </div>
        </div>
      </div>

      {/* Slideshow Section */}
      <div className="slideshow-section">
        <div className="slideshow-container">
          {images.map((img, index) => (
            <div className={`slide ${index === currentSlide ? "active" : ""}`} key={index}>
              <img src={img} alt={`Slide ${index + 1}`} />
            </div>
          ))}

          <span className="prev" onClick={prevSlide}>&#10094;</span>
          <span className="next" onClick={nextSlide}>&#10095;</span>

          <div className="dots">
            {images.map((_, index) => (
              <span key={index} className={`dot ${index === currentSlide ? "active" : ""}`}
                onClick={() => goToSlide(index)} />
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="features-section">
        <h2>Key Features of Respiguard AI</h2>
        <div className="features-grid">
          <div className="feature-tile">
            <div className="feature-icon">🧠</div>
            <h3>AI-Powered Respiratory Analysis</h3>
            <p>Advanced deep learning models for detecting respiratory diseases from mine worker data.</p>
          </div>

          <div className="feature-tile">
            <div className="feature-icon">⏱</div>
            <h3>Real-Time Monitoring</h3>
            <p>Live health scans and instant reporting of respiratory risks in mining sites.</p>
          </div>

          <div className="feature-tile">
            <div className="feature-icon">📊</div>
            <h3>Historical Trends & Analytics</h3>
            <p>Visualized reports, data-driven insights and infection trends over time.</p>
          </div>

          <div className="feature-tile">
            <div className="feature-icon">🔒</div>
            <h3>Multi-Level Access & Security</h3>
            <p>Secure access for doctors, workers, admins & labs with strict data privacy.</p>
          </div>
        </div>
      </div>

      {/* Analytics Section */}
      <div className="analytics-section">
        <h2>Respiratory Infection Trends</h2>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={[
              { month: 'Jan', infected: 5 },
              { month: 'Feb', infected: 9 },
              { month: 'Mar', infected: 12 },
              { month: 'Apr', infected: 7 },
              { month: 'May', infected: 15 },
              { month: 'Jun', infected: 10 }
            ]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="infected" stroke="#36bfe3" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Team Section */}
      <div className="team-section">
        <h2>Meet The Team</h2>
        <div className="team-container">
        <div className="team-cards-container">
          <div className="team-card">
            <div className="card-content">
              <img src="/images/ayush.jpg" alt="Ayush Ranjan Ojha" />
              <h3 className="card-title">Ayush Ranjan Ojha</h3>
              <p className="card-para">Frontend Developer</p>
            </div>
          </div>
          <div className="team-card">
            <div className="card-content">
              <img src="/images/bauxite1.jpg" alt="Prateek Sahoo" />
              <h3 className="card-title">Prateek Sahoo</h3>
              <p className="card-para">Backend Developer</p>
            </div>
          </div>
          <div className="team-card">
            <div className="card-content">
              <img src="/images/anurag.jpg" alt="Anurag Pradhan" />
              <h3 className="card-title">Anurag Pradhan</h3>
              <p className="card-para">DL Expert</p>
            </div>
          </div>
        </div>
        </div>
      </div>

      <Footer />
    </>
  );
}

export default LandingPage;
