import React, { useEffect, useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "../styles/MyProfile.css"; // you'll define the layout here
import doctorData from "../DoctorData.json";

const MyProfile = () => {
  const [doctor, setDoctor] = useState(null);

  useEffect(() => {
    const loggedInUsername = localStorage.getItem("doctorUsername"); // should match loginData.json username
    const matchedDoctor = doctorData.find(doc => doc.username === loggedInUsername);
    setDoctor(matchedDoctor);
  }, []);

  if (!doctor) {
    return (
      <div className="dashboard-container">
        <Header />
        <div className="main-section">
          <Sidebar />
          <div className="dashboard-content">
            <h2>My Profile</h2>
            <p><strong>Doctor not found.</strong></p>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content profile-content">
          <h2 className="profile-heading">My Profile</h2>
          <div className="profile-box">
            <div className="profile-left">
              <h3>{doctor.name}</h3>
              <p><strong>Email:</strong> {doctor.email}</p>
              <p><strong>Phone:</strong> {doctor.phone}</p>
              <p><strong>Qualification:</strong> {doctor.qualification}</p>
              <p><strong>Specialization:</strong> {doctor.specialization}</p>
              <p><strong>Experience:</strong> {doctor.experience}</p>
              <p><strong>Patients Handled:</strong> {doctor.patientsHandled}</p>
              <p><strong>Rating:</strong> ⭐ {doctor.rating}</p>
              <p><strong>Availability:</strong> {doctor.availability}</p>
              <p><strong>Hospital:</strong> {doctor.hospital}</p>
              <p><strong>Address:</strong> {doctor.address}</p>
            </div>
            <div className="profile-right">
              <img
                src={`/images/${doctor.image}`}
                alt={doctor.name}
                className="profile-image"
              />
              <div className="profile-icons">
                <a href={`mailto:${doctor.email}`}><img src="/icons/gmail.svg" alt="Gmail" /></a>
                <a href={`tel:${doctor.phone}`}><img src="/icons/call.svg" alt="Call" /></a>
                <a href={`https://wa.me/${doctor.phone.replace('+', '')}`} target="_blank" rel="noreferrer">
                  <img src="/icons/whatsapp.svg" alt="WhatsApp" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default MyProfile;
