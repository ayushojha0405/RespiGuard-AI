import React, { useEffect, useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/Dashboard.css";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

const useCountUp = (target, duration = 1500) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = null;
    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const percentage = Math.min(progress / duration, 1);
      const eased = Math.floor(percentage * target);
      setCount(eased);
      if (percentage < 1) {
        requestAnimationFrame(step);
      } else {
        setCount(target); // ensure it ends exactly at target
      }
    };

    requestAnimationFrame(step);
  }, [target, duration]);

  return count;
};

const Dashboard = () => {
  const doctorUsername = localStorage.getItem("doctorUsername");

  const totalPatients = useCountUp(450);
  const activeScans = useCountUp(120);
  const criticalCases = useCountUp(15);
  const recovered = useCountUp(310);

  const lineData = [
    { month: "Jan", patients: 10 },
    { month: "Feb", patients: 30 },
    { month: "Mar", patients: 20 },
    { month: "Apr", patients: 50 },
    { month: "May", patients: 40 },
  ];

  const pieData = [
    { name: "Healthy", value: 60 },
    { name: "At Risk", value: 25 },
    { name: "Infected", value: 15 },
  ];

  const COLORS = ["#1ca6c9", "#36bfe3", "#105c70"];

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1 className="animated-gradient-text">Hello, Dr. {doctorUsername}</h1>

          <div className="summary-cards">
            <div className="card">
              <h3>Total Patients</h3>
              <p>{totalPatients}</p>
            </div>
            <div className="card">
              <h3>Active Scans</h3>
              <p>{activeScans}</p>
            </div>
            <div className="card">
              <h3>Critical Cases</h3>
              <p>{criticalCases}</p>
            </div>
            <div className="card">
              <h3>Recovered</h3>
              <p>{recovered}</p>
            </div>
          </div>

          <div className="charts-section">
            <div className="chart-card">
              <h3>Monthly Patient Scans</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={lineData}>
                  <Line type="monotone" dataKey="patients" stroke="#16819c" strokeWidth={3} />
                  <CartesianGrid stroke="#ccc" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-card">
              <h3>Respiratory Health Status</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Dashboard;
