import React from 'react';
import '../styles/LandingPage.css';
import { useNavigate } from 'react-router-dom';
import Header from "./Header";
import Footer from "./Footer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';


function LandingPage() {

    const navigate = useNavigate();
    const handleLoginClick = () => {
        navigate("/login");
    };
    return (
    <>
        <Header />

      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
            <div className="hero-text">
                <h2>RespiGuard AI</h2>
                <p>AI-powered Respiratory Health Monitoring for NALCO Bauxite Mines</p>

                <div className="main-aims">
                    <p>• Early detection of respiratory diseases.</p>
                    <p>• Protect workers in hazardous mining environments.</p>
                    <p>• Provide real-time health analytics to management.</p>
                </div>

                <button className="login-button" onClick={handleLoginClick}>Start Scan / Login</button>
            </div>

            <div className="hero-image">
                <img src="/images/logo.jpg" alt="Lungs Analysis"/>
            </div>
        </div>

        {/* Slideshow Section */}
        <div className="slideshow-section">
            <div className="slideshow-container">
                <div className="slideshow">
                    <div className="slide"><img src="/images/bauxite1.jpg" alt="Mine 1" /></div>
                    <div className="slide"><img src="/images/bauxite2.jpg" alt="Mine 2" /></div>
                    <div className="slide"><img src="/images/bauxite3.jpg" alt="Mine 3" /></div>
                    <div className="slide"><img src="/images/bauxite4.jpg" alt="Mine 4" /></div>
                    <div className="slide"><img src="/images/bauxite5.jpg" alt="Mine 5" /></div>
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
        <div className="team-cards">
            <div className="card">
                <img src="/images/ayush.jpg" alt="Frontend Developer" />
                <h3>Ayush Ranjan Ojha</h3>
                <p>Frontend Developer</p>
            </div>

            <div className="card">
                <img src="/images/bauxite1.jpg" alt="Backend Developer" />
                <h3>Prateek Sahoo</h3>
                <p>Backend Developer</p>
            </div>

            <div className="card">
                <img src="/images/anurag.jpg" alt="DL Expert" />
                <h3>Anurag Pradhan</h3>
                <p>DL Expert</p>
            </div>
        </div>
        </div>

        <Footer />

    </div>
    </>
  );
}

export default LandingPage;
