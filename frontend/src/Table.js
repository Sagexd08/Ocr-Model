import React, { useState } from 'react';
import { Table as MuiTable, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';

function Table({ table }) {
  const [data, setData] = useState(table.data);

  const handleCellChange = (e, rowIndex, cellIndex) => {
    const newData = [...data];
    newData[rowIndex][cellIndex] = e.target.innerText;
    setData(newData);
  };

  return (
    <TableContainer component={Paper}>
      <MuiTable>
        <TableHead>
          <TableRow>
            {data[0].map((_, i) => (
              <TableCell key={i}>Column {i + 1}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, i) => (
            <TableRow key={i}>
              {row.map((cell, j) => (
                <TableCell
                  key={j}
                  contentEditable
                  onBlur={(e) => handleCellChange(e, i, j)}
                >
                  {cell}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </MuiTable>
    </TableContainer>
  );
}

export default Table;