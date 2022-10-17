import pandas as pd

from ..tournament import *

def test_get_teams_df():
    teams_df = get_teams_data()
    assert isinstance(teams_df, pd.DataFrame)
    assert(len(teams_df) > 0)
    
def test_get_fixture_df():
    fix_df = get_fixture_data()
    assert isinstance(fix_df, pd.DataFrame)
    assert(len(fix_df) > 0)


def test_find_group():
    teams_df = get_teams_data()
    assert find_group("Qatar", teams_df) == "A"    
    assert find_group("England", teams_df) == "B"


def test_find_group_bad_team():
    teams_df = get_teams_data()
    assert find_group("Italy",teams_df) == None


def test_create_tournament():
    t = Tournament()
    assert isinstance(t.teams_df, pd.DataFrame)
    assert isinstance(t.fixtures_df, pd.DataFrame)    
    
    
