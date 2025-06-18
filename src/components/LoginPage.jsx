import React, { useState } from "react";
import "./../styles/LoginPage.css";
import Header from "./Header";
import Footer from "./Footer";
import { useNavigate } from "react-router-dom";
import loginData from "../loginData.json";

const LoginPage = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    const foundUser = loginData.find(
      (user) => user.username === username && user.password === password
    );

    if (foundUser) {
      localStorage.setItem("doctorUsername", foundUser.fullname);
      navigate("/dashboard", { state: { doctorUsername: foundUser.fullname } });
      const goHome = () => {
        navigate("/");
      };
    } else {
      alert("Invalid credentials");
    }
  };

  return (
    <>
      <Header />
      <div className="login-container">
        <div><h2>Doctor Login</h2></div>
        <form onSubmit={handleSubmit}>
          <div><input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          /></div>
          <div><input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          /></div>
          <br />
          <button type="submit">Login</button>
        </form>
        {error && <p className="error">{error}</p>}
      </div>
      <Footer />

    </>
  );
};

export default LoginPage;
