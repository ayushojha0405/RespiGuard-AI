import React, { useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/TestLab.css";

const TestLab = () => {
  const doctorUsername = localStorage.getItem("doctorUsername");
  const [testFile, setTestFile] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (testFile) {
      alert(`File "${testFile.name}" submitted for AI test.`);
      setTestFile(null);
    } else {
      alert("Please upload a JPG file.");
    }
  };

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Test Lab - Dr. {doctorUsername}</h1>

          <form className="testlab-form" onSubmit={handleSubmit}>
            <div className="form-section">
              <h3>Upload Chest Scan (JPG Only)</h3>
              <input
                type="file"
                accept=".jpg"
                onChange={(e) => setTestFile(e.target.files[0])}
                required
              />
              {testFile && <p>Selected File: {testFile.name}</p>}
            </div>

            <button type="submit" className="submit-btn">
              Test AI Model
            </button>
          </form>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default TestLab;
