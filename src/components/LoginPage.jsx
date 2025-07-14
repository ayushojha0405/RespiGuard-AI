import React, { useState, useEffect } from "react";
import "./../styles/LoginPage.css";
import Header from "./Header";
import Footer from "./Footer";
import { useNavigate } from "react-router-dom";
import loginData from "../loginData.json";

const LoginPage = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [captcha, setCaptcha] = useState("");
  const [captchaInput, setCaptchaInput] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  // Generate CAPTCHA on component mount
  useEffect(() => {
    generateCaptcha();
  }, []);

  const generateCaptcha = () => {
    const random = Math.floor(10000 + Math.random() * 90000).toString();
    setCaptcha(random);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (captchaInput !== captcha) {
      setError("Incorrect CAPTCHA");
      generateCaptcha();
      return;
    }

    const foundUser = loginData.find(
      (user) => user.username === username && user.password === password
    );

    if (foundUser) {
      localStorage.setItem("doctorUsername", foundUser.username);
      navigate("/dashboard", { state: { doctorUsername: foundUser.fullname } });
    } else {
      alert("Invalid credentials");
    }
  };

  return (
    <>
      <Header />
      <div className="login-container">
        <h2>Doctor Login</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <div className="captcha-container">
            <div className="captcha-display">{captcha}</div>
            <input
              type="text"
              placeholder="Enter Captcha"
              value={captchaInput}
              onChange={(e) => setCaptchaInput(e.target.value)}
              required
            />
          </div>

          <button type="submit">Login</button>
        </form>

        {error && <p className="error">{error}</p>}
      </div>
      <Footer />
    </>
  );
};

export default LoginPage;
