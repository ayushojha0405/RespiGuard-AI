import React from "react";
import "./../styles/Footer.css";

const Footer = () => {
  return (
    <div className="footer">
      <div className="footer-section">
        <h3>RespiGuard AI</h3>
        <p>AI-powered respiratory health monitoring for bauxite miners.</p>
      </div>

      <div className="footer-section">
        <h4>Contact Us</h4>
        <p>Email: support@respiguardai.com</p>
        <p>Phone: +91-9876543210</p>
      </div>

      <div className="footer-section">
        <h4>In Collaboration with</h4>
        <p>NALCO (National Aluminium Company Limited)</p>
      </div>

      <div className="footer-bottom">
        <p>© 2025 RespiGuard AI. All Rights Reserved.</p>
      </div>
    </div>
  );
};

export default Footer;
