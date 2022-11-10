import React from 'react';
import Table from 'react-bootstrap/Table';

const Fixture = (props) => {
   return (
    <Table striped bordered hover>
        <tbody>
            <tr>
            <th>{props.date}</th>
            <th>{props.stage}</th>
            <th>{props.team_1} vs {props.team_2}</th>
            <th>{props.session}</th>
            </tr>
        </tbody>
    </Table>
   );
};
 
export default Fixture;