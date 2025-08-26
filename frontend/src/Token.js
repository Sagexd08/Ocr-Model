import React, { useState } from 'react';

function Token({ token }) {
  const [text, setText] = useState(token.text);

  const handleTextChange = (e) => {
    setText(e.target.innerText);
  };

  return (
    <span
      contentEditable
      onBlur={handleTextChange}
      style={{ border: '1px solid black', margin: '2px', padding: '2px' }}
    >
      {text}
    </span>
  );
}

export default Token;