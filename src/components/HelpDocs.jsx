import React, { useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";
import "./../styles/HelpDocs.css";

const faqs = [
  { question: "What file types are supported?", answer: "Only .jpg images are currently supported." },
  { question: "Can I edit a submitted report?", answer: "No, please contact support for critical edits." },
  { question: "Why is dashboard not updating?", answer: "Try refreshing the page or check your internet connection." },
  { question: "Where is the history stored?", answer: "In a local JSON file (or eventually backend). Visible in the History tab." },
  { question: "How secure is the data?", answer: "Data is locally stored for now. In future, it will be server-side encrypted." },
  { question: "What is the purpose of Test Lab?", answer: "To test how the AI model works without storing patient data." },
  { question: "Supported Browsers?", answer: "Chrome, Firefox, Edge, Safari (latest versions)." },
  { question: "Can I use on mobile?", answer: "Yes, but desktop is recommended for full layout & charts." },
  { question: "Need help?", answer: "Email us at support@respiguardai.com." },
];

const HelpDocs = () => {
  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="dashboard-container">
      <Header />
      <div className="main-section">
        <Sidebar />
        <div className="dashboard-content">
          <h1>Help & Documentation</h1>

          <div className="help-section">
            <h2>FAQs</h2>
            <div className="faq-container">
              {faqs.map((faq, index) => (
                <div key={index} className="faq-item">
                  <button className="faq-question" onClick={() => toggleFAQ(index)}>
                    {faq.question}
                    <span className="arrow">{openIndex === index ? "▲" : "▼"}</span>
                  </button>
                  <div
                    className={`faq-answer-wrapper ${openIndex === index ? "open" : ""}`}
                  >
                    <div className="faq-answer">{faq.answer}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Docs section can follow here */}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default HelpDocs;
