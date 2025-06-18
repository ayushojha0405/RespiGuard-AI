import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // <-- Add this
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/ReportScan.css";

const ReportScan = () => {
  const doctorUsername = localStorage.getItem("doctorUsername");
  const navigate = useNavigate(); // <-- useNavigate hook

  const [patientId, setPatientId] = useState("");
  const [patientName, setPatientName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [examDate, setExamDate] = useState(new Date().toISOString().substr(0, 10));
  const [scanFile, setScanFile] = useState(null);
  const [submitted, setSubmitted] = useState(false); // <-- to control button visibility

  const handleSubmit = async (e) => {
    e.preventDefault();

    const toBase64 = (file) =>
      new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = (err) => reject(err);
      });

    const scanImageBase64 = scanFile ? await toBase64(scanFile) : null;

    const newReport = {
      patientId,
      patientName,
      age,
      gender,
      examDate,
      doctorName: doctorUsername,
      scanFile: scanFile ? scanFile.name : null,
      scanImageData: scanImageBase64,
    };

    const existingReports = JSON.parse(localStorage.getItem("details.json")) || [];
    const updatedReports = [...existingReports, newReport];
    localStorage.setItem("details.json", JSON.stringify(updatedReports));

    alert(`Report Submitted for Patient ID: ${patientId}`);
    setSubmitted(true); // <-- enable "See Submitted Data" button

    // Reset form
    setPatientId("");
    setPatientName("");
    setAge("");
    setGender("");
    setExamDate(new Date().toISOString().substr(0, 10));
    setScanFile(null);
  };

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Report Scan - Dr. {doctorUsername}</h1>

          <form className="report-form" onSubmit={handleSubmit}>
            <div className="form-section">
              <h3>Patient Information</h3>

              <input
                type="text"
                placeholder="Patient ID"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                required
              />
              <input
                type="text"
                placeholder="Patient Name"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                required
              />
              <input
                type="number"
                placeholder="Age"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                required
              />
              <select value={gender} onChange={(e) => setGender(e.target.value)} required>
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
              <input
                type="date"
                value={examDate}
                onChange={(e) => setExamDate(e.target.value)}
                required
              />
              <input type="text" value={doctorUsername} readOnly />
            </div>

            <div className="form-section">
              <h3>Upload Scan Image (JPG Only)</h3>
              <input
                type="file"
                accept=".jpg"
                onChange={(e) => setScanFile(e.target.files[0])}
                required
              />
              {scanFile && <p>Selected File: {scanFile.name}</p>}
            </div>

            <button type="submit" className="submit-btn">
              Submit Report
            </button>
          </form>

          {submitted && (
            <button onClick={() => navigate("/Check")} className="submit-btn" style={{ marginTop: "20px" }}>
              See Submitted Data
            </button>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default ReportScan;
