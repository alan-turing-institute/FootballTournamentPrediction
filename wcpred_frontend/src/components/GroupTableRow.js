import React from 'react';

const GroupTableRow = ({row}) => {
    return (
        <tr>
        <th>{row.position}</th>
        <th>{row.team}</th>
        <th>{row.points}</th>
        <th>{row.gs}</th>
        <th>{row.ga}</th>
        </tr>       
    );
};

export default GroupTableRow;
