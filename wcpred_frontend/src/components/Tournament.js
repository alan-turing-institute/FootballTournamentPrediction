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
 
    // useEffect to update group tables
    const [isLoadingGroups, setLoadingGroups] = useState(true);
    const [groupData, setGroupData] = useState({});
    useEffect(() => {
        const url = getBaseURL() + "/groups";
        if (isLoadingGroups) {
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
                setLoadingGroups(false);
            });
        }
    }, [isLoadingGroups]);

    // useEffect to update next fixture
    const [nextFixture, setNextFixture] = useState({});
    const [isLoadingFixture, setLoadingFixture] = useState(true);
    useEffect(() => {
        const url = getBaseURL() + "/matches/next";
        if (isLoadingFixture) {
            const fetchData = async () => {
                try {
                  const response = await fetch(url);
                  const json = await response.json();
                  console.log(json);
                  setNextFixture(json);
                } catch (error) {
                  console.log("error fetching fixture data", error);
                }
            };
            fetchData().then(() => {
                setLoadingFixture(false);
            });
        }
    }, [isLoadingFixture]);

    // play next match
    const [latestResult, setLatestResult] = useState({});
    const [isLoadingResult, setLoadingResult] = useState(true);
    function playMatch() {
        console.log("Play match!");
        const url = getBaseURL() + "/matches/next";
        if (isLoadingResult) {
            const fetchData = async () => {
                try {
                const response = await fetch(url, {method: 'POST'});
                const json = await response.json();
                console.log(json);
                setLatestResult(json);
                } catch (error) {
                console.log("error fetching result data", error);
                }
            };
            fetchData().then(() => {
                setLoadingResult(false);
            });
        }
    };



    const handleMatchClick = () => playMatch();
    return (
        <div>
            <div>
            { isLoadingGroups ? "loading..." : (
                Object.entries(groupData).map(([k,v]) => { 
                    return (
                        <GroupTable groupName={k} groupData={v} />
                    )
                })
            )}
            </div>
            <div>
                Next fixture:
            { isLoadingFixture ? "loading..." : (
                <Fixture 
                date={nextFixture.date}
                stage={nextFixture.stage}
                team_1={nextFixture.team_1}
                team_2={nextFixture.team_2}
                session={nextFixture.sessionid}
                />
            )}                
            </div>
        
            <Button 
            variant="primary"
            disabled={isLoadingFixture}
            onClick={!isLoadingFixture ? handleMatchClick : null}
            >
            {isLoadingFixture ? 'Loading…' : 'Play next match'}
            </Button>
            <div>
                Latest result:
            
            </div>
        </div>
    );
};
 
export default Tournament;