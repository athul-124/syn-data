import React, { useMemo } from 'react';
import { useTable, usePagination } from 'react-table';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const DataPreview = ({ data, title = "Data Preview" }) => {
  const columns = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    return Object.keys(data[0]).map(key => ({
      Header: key,
      accessor: key,
      Cell: ({ value }) => (
        <div className="truncate max-w-xs" title={value}>
          {value}
        </div>
      )
    }));
  }, [data]);
  
  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    page,
    prepareRow,
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    state: { pageIndex }
  } = useTable(
    {
      columns,
      data: data || [],
      initialState: { pageIndex: 0, pageSize: 10 }
    },
    usePagination
  );
  
  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg border p-8 text-center">
        <p className="text-gray-500">No data to preview</p>
      </div>
    );
  }
  
  return (
    <div className="bg-white rounded-lg border">
      <div className="px-6 py-4 border-b">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        <p className="text-sm text-gray-600">{data.length} rows</p>
      </div>
      
      <div className="overflow-x-auto">
        <table {...getTableProps()} className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            {headerGroups.map(headerGroup => (
              <tr {...headerGroup.getHeaderGroupProps()}>
                {headerGroup.headers.map(column => (
                  <th
                    {...column.getHeaderProps()}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    {column.render('Header')}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody {...getTableBodyProps()} className="bg-white divide-y divide-gray-200">
            {page.map(row => {
              prepareRow(row);
              return (
                <tr {...row.getRowProps()}>
                  {row.cells.map(cell => (
                    <td
                      {...cell.getCellProps()}
                      className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                    >
                      {cell.render('Cell')}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      {pageCount > 1 && (
        <div className="px-6 py-3 border-t flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => previousPage()}
              disabled={!canPreviousPage}
              className="p-2 rounded-md border disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => nextPage()}
              disabled={!canNextPage}
              className="p-2 rounded-md border disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
            >
              <ChevronRight size={16} />
            </button>
          </div>
          
          <span className="text-sm text-gray-700">
            Page {pageIndex + 1} of {pageOptions.length}
          </span>
        </div>
      )}
    </div>
  );
};

export default DataPreview;