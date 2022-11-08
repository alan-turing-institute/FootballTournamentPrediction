import { async } from 'q';
import React, { useEffect, useState } from 'react';
import Button from 'react-bootstrap/Button';
import Fixture from './Fixture';
import GroupTable from './GroupTable';
import getBaseURL from './utils';

function simulateNetworkRequest() {
    return new Promise((resolve) => setTimeout(resolve, 2000));
} 

const Tournament = (props) => {
    const [isLoading, setLoading] = useState(true);
    const [groupData, setGroupData] = useState({});

    useEffect(() => {
        const url = getBaseURL() + "/groups";
        if (isLoading) {
            const fetchData = async () => {
                try {
                  const response = await fetch(url);
                  const json = await response.json();
                  console.log(json);
                  setGroupData(json);
                } catch (error) {
                  console.log("error fetching group data", error);
                }
            };
          
            fetchData().then(() => {
                setLoading(false);
            });
        }
    }, [isLoading]);

    const handleClick = () => setLoading(true);
    return (
        <div>
            { isLoading ? "loading..." : (
                //<GroupTable groupName="A" groupData={groupData.A} />
                Object.entries(groupData).map(([k,v]) => { 
                    return (
                        <GroupTable groupName={k} groupData={v} />
                    )
                })
            )}

        
            <Button 
            variant="primary"
            disabled={isLoading}
            onClick={!isLoading ? handleClick : null}
            >
            {isLoading ? 'Loading…' : 'Click to run tournament'}
            </Button>
        </div>
    );
};
 
export default Tournament;