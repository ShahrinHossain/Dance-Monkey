import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar.js';
import '../styles/SearchResults.css';

function SearchResults() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const query = searchParams.get('q');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Replace this URL with your backend API endpoint
    const fetchResults = async () => {
      setLoading(true);
      try {
        // Example API call - replace with your backend URL
        const response = await fetch(`http://localhost:5000/api/search?q=${query}`);
        const data = await response.json();
        setResults(data.results || []);
      } catch (error) {
        console.error('Error fetching results:', error);
        // Mock data for demonstration
        setResults([
          { id: 1, title: 'Result 1', description: 'This is a description for result 1', url: 'https://example.com/1' },
          { id: 2, title: 'Result 2', description: 'This is a description for result 2', url: 'https://example.com/2' },
          { id: 3, title: 'Result 3', description: 'This is a description for result 3', url: 'https://example.com/3' },
        ]);
      } finally {
        setLoading(false);
      }
    };

    if (query) {
      fetchResults();
    }
  }, [query]);

  const handleResultClick = (id) => {
    navigate(`/detail/${id}`);
  };

  return (
    <div className="results-container">
      <header className="results-header">
        <h2 className="logo-small" onClick={() => navigate('/')}>Search Engine</h2>
        <SearchBar initialQuery={query} isResultsPage={true} />
      </header>

      <div className="results-content">
        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <>
            <p className="results-stats">About {results.length} results for "{query}"</p>
            <div className="results-list">
              {results.map((result) => (
                <div
                  key={result.id}
                  className="result-card"
                  onClick={() => handleResultClick(result.id)}
                >
                  <h3 className="result-title">{result.title}</h3>
                  <p className="result-url">{result.url}</p>
                  <p className="result-description">{result.description}</p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default SearchResults;