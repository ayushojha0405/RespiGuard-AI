import React from "react";
import { useNavigate } from "react-router-dom";
import "./../styles/Header.css";

const Header = () => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate("/");
  };

  return (
    <div className="titlebar">
      <div className="logo-left" onClick={handleLogoClick} style={{ cursor: "pointer" }}>
        <img src="/images/logo.jpg" alt="RespiGuard AI Logo" />
      </div>
      <div className="title-center">
        <h1>RespiGuard AI</h1>
      </div>
      <div className="logo-right">
        <img src="/images/nalco.jpg" alt="NALCO Logo" />
      </div>
    </div>
  );
};

export default Header;
