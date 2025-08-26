import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Review from './Review';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/review/:jobId" element={<Review />} />
      </Routes>
    </Router>
  );
}

export default App;