import React from 'react';
import Token from './Token';
import Table from './Table';

function Page({ page }) {
  return (
    <div>
      <h2>Page {page.page_number}</h2>
      <div>
        {page.tokens.map((token, i) => (
          <Token key={i} token={token} />
        ))}
      </div>
      <div>
        {page.tables.map((table, i) => (
          <Table key={i} table={table} />
        ))}
      </div>
    </div>
  );
}

export default Page;