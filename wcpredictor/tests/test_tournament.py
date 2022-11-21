import random

import numpy as np
import pandas as pd

from wcpredictor import (
    Group,
    Tournament,
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
    g.calc_standings()

    np.testing.assert_array_equal(g.standings[:, 0], np.array([3, 4, 1, 2]))


def test_many_standings():
    """
    Define some sanity checks, run the group with random results many times.
    """

    def sanity_check_points(g):
        top_points = g.table[g.standings["1st"]]["points"]
        if (
            g.table[g.standings["2nd"]]["points"] > top_points
            or g.table[g.standings["3rd"]]["points"] > top_points
            or g.table[g.standings["4th"]]["points"] > top_points
        ):
            return False
        return True

    def sanity_check_goal_diff(g):

        top_points = g.table[g.standings["1st"]]["points"]
        top_gd = g.table[g.standings["1st"]]["goal_difference"]
        if (
            g.table[g.standings["2nd"]]["points"] == top_points
            and g.table[g.standings["2nd"]]["goal_difference"] > top_gd
        ):
            return False
        if (
            g.table[g.standings["3rd"]]["points"] == top_points
            and g.table[g.standings["3rd"]]["goal_difference"] > top_gd
        ):
            return False
        if (
            g.table[g.standings["4th"]]["points"] == top_points
            and g.table[g.standings["4th"]]["goal_difference"] > top_gd
        ):
            return False
        return True

    for _ in range(100):
        g = Group("A", ["Qatar", "Ecuador", "Senegal", "Netherlands"])
        g.results.append(
            {"Ecuador": random.randint(0, 4), "Qatar": random.randint(0, 4)}
        )
        g.results.append(
            {"Netherlands": random.randint(0, 4), "Qatar": random.randint(0, 4)}
        )
        g.results.append(
            {"Netherlands": random.randint(0, 4), "Senegal": random.randint(0, 4)}
        )
        g.results.append(
            {"Netherlands": random.randint(0, 4), "Ecuador": random.randint(0, 4)}
        )
        g.results.append(
            {"Senegal": random.randint(0, 4), "Qatar": random.randint(0, 4)}
        )
        g.results.append(
            {"Ecuador": random.randint(0, 4), "Senegal": random.randint(0, 4)}
        )
        g.calc_standings()
        assert sanity_check_points(g)
        assert sanity_check_goal_diff(g)


def test_play_group_stage(mocker):
    """
    Play simulated group stage 100 times - ensure that we always have 16 qualifying
    teams at the end.
    """

    def pick_random_score():
        s1 = random.randint(0, 3)
        s2 = random.randint(0, 3)
        return s1, s2

    for _ in range(100):
        t = Tournament()
        mocker.patch(
            "wcpredictor.src.tournament.predict_group_match",
            return_value=pick_random_score(),
        )
        t.play_group_stage("dummy")
        for group in t.groups.values():
            assert len(group.results) == 6
            assert len(group.get_qualifiers()) == 2


def test_play_knockout_stages(mocker):
    """
    Play simulated knockout stages 100 times, check that we always get a winner.
    """

    def pick_random_score():
        s1 = random.randint(0, 3)
        s2 = random.randint(0, 3)
        return s1, s2

    def pick_random_winner(wc_pred, team_1, team_2, seed):
        if random.random() > 0.5:
            return team_1
        else:
            return team_2

    teams_df = get_teams_data()
    teams = list(teams_df.Team.values)
    t = Tournament()
    mocker.patch(
        "wcpredictor.src.tournament.predict_group_match",
        return_value=pick_random_score(),
    )
    t.play_group_stage("dummy")
    for _ in range(100):
        mocker.patch(
            "wcpredictor.src.tournament.predict_knockout_match", new=pick_random_winner
        )
        t.play_knockout_stages("dummy")
        assert t.winner in teams
