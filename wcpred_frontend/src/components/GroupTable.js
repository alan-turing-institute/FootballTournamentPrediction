import React from 'react';
import { propTypes } from 'react-bootstrap/esm/Image';
import Table from 'react-bootstrap/Table';

import GroupTableRow from './GroupTableRow';

const GroupTable = (props) => {
    return (
        <div>
            <header>
                <h1>Group {props.groupName} </h1>
            </header>
            <Table striped bordered hover>
                <thead>
                    <tr>
                        <th>Position</th>
                        <th>Team</th>
                        <th>Points</th>
                        <th>Goals Scored</th>
                        <th>Goals Against</th>
                    </tr>
                </thead>
                <tbody>
                    {props.groupData.map(row => {
                        return (
                            <GroupTableRow row={row} />
                        )
                    })}
                </tbody>
                
            </Table>
        </div>
   );
};
 
export default GroupTable;