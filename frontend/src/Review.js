import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import Page from './Page';

function Review() {
  const [document, setDocument] = useState(null);
  const { jobId } = useParams();

  useEffect(() => {
    fetch(`/api/review/${jobId}`)
      .then(res => res.json())
      .then(data => setDocument(data));
  }, [jobId]);

  const handleSave = () => {
    fetch(`/api/review/${jobId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(document)
      }
    );
  };

  if (!document) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h1>Review Document</h1>
      {document.pages.map(page => (
        <Page key={page.page_number} page={page} />
      ))}
      <button onClick={handleSave}>Save</button>
    </div>
  );
}

export default Review;
