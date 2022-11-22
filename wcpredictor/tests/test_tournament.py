import random

import numpy as np
import pandas as pd

from wcpredictor import (
    Group,
    Tournament,
    WCPred,
    get_fixture_data,
    get_teams_data,
    sort_teams_by,
)


def test_get_teams_df():
    teams_df = get_teams_data()
    assert isinstance(teams_df, pd.DataFrame)
    assert len(teams_df) > 0


def test_get_fixture_df():
    fix_df = get_fixture_data()
    assert isinstance(fix_df, pd.DataFrame)
    assert len(fix_df) > 0


def test_sort_by_points():
    test_dict = {
        "Qatar": {
            "points": 9,
            "goal_difference": 5,
            "goals_for": 6,
            "goals_against": 1,
        },
        "Senegal": {
            "points": 6,
            "goal_difference": 2,
            "goals_for": 4,
            "goals_against": 2,
        },
        "Ecuador": {
            "points": 3,
            "goal_difference": -2,
            "goals_for": 2,
            "goals_against": 4,
        },
        "Netherlands": {
            "points": 0,
            "goal_difference": -5,
            "goals_for": 1,
            "goals_against": 6,
        },
    }
    sorted_list = sort_teams_by(test_dict, "points")
    expected_team_order = ["Qatar", "Senegal", "Ecuador", "Netherlands"]
    assert [t["team"] for t in sorted_list] == expected_team_order


def test_sort_by_goal_difference():
    test_dict = {
        "Qatar": {
            "points": 4,
            "goal_difference": -3,
            "goals_for": 3,
            "goals_against": 6,
        },
        "Senegal": {
            "points": 4,
            "goal_difference": 3,
            "goals_for": 6,
            "goals_against": 3,
        },
        "Ecuador": {
            "points": 4,
            "goal_difference": 2,
            "goals_for": 4,
            "goals_against": 2,
        },
        "Netherlands": {
            "points": 4,
            "goal_difference": -2,
            "goals_for": 2,
            "goals_against": 4,
        },
    }
    sorted_list = sort_teams_by(test_dict, "goal_difference")
    expected_team_order = ["Senegal", "Ecuador", "Netherlands", "Qatar"]
    assert [t["team"] for t in sorted_list] == expected_team_order


def test_create_tournament():
    t = Tournament()
    assert isinstance(t.teams_df, pd.DataFrame)
    assert isinstance(t.fixtures_df, pd.DataFrame)
    assert len(t.groups) == 8
    assert set(t.groups.keys()) == {"A", "B", "C", "D", "E", "F", "G", "H"}


def test_create_group():
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    assert isinstance(g, Group)


def test_table():
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(["Ecuador", "Netherlands"]),
        "away_team": np.array(["Qatar", "Qatar"]),
        "home_score": np.array([3, 2]).reshape((2, 1)),
        "away_score": np.array([1, 2]).reshape((2, 1)),
    }
    g.add_results(results)
    g.calc_table()

    np.testing.assert_array_equal(g.table["points"][:, 0], np.array([1, 3, 0, 1]))
    np.testing.assert_array_equal(g.table["goals_for"][:, 0], np.array([3, 3, 0, 2]))
    np.testing.assert_array_equal(
        g.table["goals_against"][:, 0], np.array([5, 1, 0, 2])
    )
    np.testing.assert_array_equal(
        g.table["goal_difference"][:, 0], np.array([-2, 2, 0, 0])
    )


def test_standings_simple_case():
    """
    clear order - 1st, 2nd, 3rd, 4th have different points
    """
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.array([3, 2, 1, 0, 3, 1]).reshape((6, 1)),
        "away_score": np.array([1, 2, 1, 2, 2, 0]).reshape((6, 1)),
    }
    g.add_results(results)
    g.calc_table()
    g.calc_standings()

    # Ecuador should have 9 points, Senegal 3, Netherlands 2, Qatar 1
    np.testing.assert_array_equal(g.standings[:, 0], np.array([4, 1, 2, 3]))


def test_standings_2nd_3rd_goal_difference():
    """
    2nd and 3rd place teams have equal points but one has better goal difference
    """
    # Netherlands beat everyone, Senegal and Qatar both beat Ecuador, but
    # Senegal wins by more goals, and they draw with each other.
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.array([0, 2, 3, 2, 3, 1]).reshape((6, 1)),
        "away_score": np.array([1, 0, 1, 1, 3, 3]).reshape((6, 1)),
    }
    g.add_results(results)
    g.calc_table()
    g.calc_standings()

    np.testing.assert_array_equal(g.standings[:, 0], np.array([3, 4, 2, 1]))


def test_standings_2nd_3rd_goals_scored():
    """
    1st 2nd and 3rd place have equal points, but 1st has better goal difference,
    2nd has better goals scored
    """
    # Top 3 teams have 1 win, 1 draw, 1 loss each
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.array([1, 2, 2, 2, 2, 1]).reshape((6, 1)),
        "away_score": np.array([1, 2, 0, 2, 3, 0]).reshape((6, 1)),
    }
    g.add_results(results)
    g.calc_table()
    g.calc_standings()

    np.testing.assert_array_equal(g.standings[:, 0], np.array([2, 3, 4, 1]))


def test_standings_all_equal():
    """
    If all matches are 1-1 draws, order is random, but code should at least run
    """
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.array([1, 1, 1, 1, 1, 1]).reshape((6, 1)),
        "away_score": np.array([1, 1, 1, 1, 1, 1]).reshape((6, 1)),
    }
    g.add_results(results)
    g.calc_table()
    g.calc_standings()
    assert set(g.standings[:, 0]) == {1, 2, 3, 4}


def test_standings_higher_scoring_draws():
    """
    If all matches are draws, sort by goals scored
    """
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.array([1, 2, 1, 1, 3, 1]).reshape((6, 1)),
        "away_score": np.array([1, 2, 1, 1, 3, 1]).reshape((6, 1)),
    }
    g.add_results(results)
    g.calc_table()
    g.calc_standings()

    np.testing.assert_array_equal(g.standings[:, 0], np.array([1, 4, 2, 3]))


def test_standings_1st_2nd_head_to_head():
    """
    1st 2nd have equal points, gd, goals scored, but 1st beat second
    """
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.array([1, 4, 0, 2, 2, 0]).reshape((6, 1)),
        "away_score": np.array([1, 1, 2, 0, 3, 2]).reshape((6, 1)),
    }
    g.add_results(results)
    g.calc_table()

    for _ in range(20):
        # test could pass in error as the next tiebreaker is random if head-to-head is
        # skipped, so repeat the standings computation to verify
        g.calc_standings(head_to_head=True)
        np.testing.assert_array_equal(g.standings[:, 0], np.array([3, 4, 1, 2]))


def test_many_standings():
    """
    Define some sanity checks, run the group with random results many times.
    """
    g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
    results = {
        "home_team": np.array(
            [
                "Ecuador",
                "Netherlands",
                "Netherlands",
                "Netherlands",
                "Senegal",
                "Ecuador",
            ]
        ),
        "away_team": np.array(
            ["Qatar", "Qatar", "Senegal", "Ecuador", "Qatar", "Senegal"]
        ),
        "home_score": np.random.randint(0, 5, size=(6, 100)),
        "away_score": np.random.randint(0, 5, size=(6, 100)),
    }
    g.add_results(results)
    g.calc_table()
    g.calc_standings()
    rank_idx = g.standings.argsort(axis=0)

    for i in range(100):
        pts = g.table["points"][rank_idx[:, i], i]
        # pts (in rank order) should be >= next team's pts
        assert np.all(pts[:-1] >= pts[1:])

        top_points_mask = pts == pts.max()
        if top_points_mask.sum() == 1:
            continue  # group has clear pts winner, don't check gd

        gd = g.table["goal_difference"][rank_idx[:, i], i][top_points_mask]
        # gd (in rank order) amongst teams tied on pts should be >= next gd
        assert np.all(gd[:-1] >= gd[1:])


def test_play_group_stage(mocker):
    """
    Play simulated group stage 100 times - ensure that we always have 16 qualifying
    teams at the end.
    """

    def pick_random_score():
        fixtures_df = get_fixture_data()
        fixtures_df = fixtures_df[fixtures_df["Stage"] == "Group"]
        results = {
            "home_team": fixtures_df["Team_1"].values,
            "away_team": fixtures_df["Team_2"].values,
            "home_score": np.random.randint(0, 5, size=(len(fixtures_df), 100)),
            "away_score": np.random.randint(0, 5, size=(len(fixtures_df), 100)),
        }
        return results

    t = Tournament()
    mocker.patch.object(WCPred, "sample_score", return_value=pick_random_score())
    t.play_group_stage(
        WCPred(
            pd.DataFrame({"home_team": ["dummy"], "away_team": ["dummy"]}),
            teams=get_teams_data()["Team"].values,
        )
    )
    for group in t.groups.values():
        assert group.results["home_score"].shape == (6, 100)
        gq = group.get_qualifiers()
        assert len(gq[0]) == 100
        assert len(gq[1]) == 100


def test_play_knockout_stages(mocker):
    """
    Play simulated knockout stages 100 times, check that we always get a winner.
    """

    def pick_random_score():
        fixtures_df = get_fixture_data()
        fixtures_df = fixtures_df[fixtures_df["Stage"] == "Group"]
        results = {
            "home_team": fixtures_df["Team_1"].values,
            "away_team": fixtures_df["Team_2"].values,
            "home_score": np.random.randint(0, 5, size=(len(fixtures_df), 100)),
            "away_score": np.random.randint(0, 5, size=(len(fixtures_df), 100)),
        }
        return results

    def pick_random_winner(self, home_team, away_team, *args, **kwargs):
        return np.array(
            [
                np.random.choice([home_team[i], away_team[i]])
                for i in range(len(home_team))
            ]
        )

    t = Tournament(num_samples=100)
    mocker.patch.object(WCPred, "sample_score", return_value=pick_random_score())
    mocker.patch.object(WCPred, "sample_outcome", new=pick_random_winner)
    wc_pred = WCPred(
        pd.DataFrame({"home_team": ["dummy"], "away_team": ["dummy"]}),
        teams=get_teams_data()["Team"].values,
    )
    t.play_group_stage(wc_pred)
    t.play_knockout_stages(wc_pred)
    teams = get_teams_data()["Team"].values
    assert np.isin(t.winner, teams).all()
