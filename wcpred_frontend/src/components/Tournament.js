import React, { useEffect, useState } from 'react';
import Button from 'react-bootstrap/Button';
import Fixture from './Fixture';
import GroupTable from './GroupTable';

function simulateNetworkRequest() {
    return new Promise((resolve) => setTimeout(resolve, 2000));
} 

const Tournament = () => {
    const [isLoading, setLoading] = useState(false);

    useEffect(() => {
        if (isLoading) {
        simulateNetworkRequest().then(() => {
            setLoading(false);
        });
        }
    }, [isLoading]);

    const handleClick = () => setLoading(true);

    return (
       <Button 
        variant="primary"
        disabled={isLoading}
        onClick={!isLoading ? handleClick : null}
        >
        {isLoading ? 'Loading…' : 'Click to run tournament'}
        </Button>
   );
};
 
export default Tournament;