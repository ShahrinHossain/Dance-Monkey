import React from 'react';
import SearchBar from '../components/SearchBar.js';
import '../styles/Home.css';

function Home() {
  return (
    <div className="home-container">
      <div className="home-content">
        <h1 className="logo">Search Engine</h1>
        <SearchBar />
      </div>
    </div>
  );
}

export default Home;