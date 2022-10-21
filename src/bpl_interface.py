"""
Interface to the NumPyro team model in bpl-next:
https://github.com/anguswilliams91/bpl-next
"""
import numpy as np
import pandas as pd
from bpl import ExtendedDixonColesMatchPredictor
from typing import Optional, Union, List, Tuple
from jax.numpy import DeviceArray

class WCPred:
    def __init__(self,
                 results: pd.DataFrame,
                 fixtures: pd.DataFrame,
                 ratings: Optional[pd.DataFrame] = None,
                 teams: Optional[List[str]] = None,
                 years: Optional[List[int]] = None):
        self.results = results
        self.fixtures = fixtures
        self.ratings = ratings
        if teams is None:
            # teams is just every team that has played in results
            self.teams = list(set(self.results["home_team"]) | set(self.results["away_team"]))
        else:
            # disregard any games which involves a team that isnt in teams
            self.teams = teams
            self.results = self.results[self.results.home_team.isin(teams) &
                                        self.results.away_team.isin(teams)]
        if years is not None:
            self.results = self.results[self.results.date.dt.year.isin(years)]
        self.training_data = None
        self.model = None
        
    def get_result_dict(self) -> dict[str, np.array]:
        """
        put results into dictionary to train model
        """
        return {
            "home_team": np.array(self.results.home_team),
            "away_team": np.array(self.results.away_team),
            "home_goals": np.array(self.results.home_score),
            "away_goals": np.array(self.results.away_score),
        }

    def get_ratings_dict(self) -> dict:
        """Create a dataframe containing the fifa team ratings."""
        ratings = self.ratings[self.ratings.Team.isin(self.teams)]
        ratings_dict = {
            row.Team: np.array([row.Attack, row.Midfield, row.Defence, row.Overall])
            for index,row in ratings.iterrows()
        }
        if len(ratings_dict) != len(self.teams):
            raise ValueError(
                f"Must have FIFA ratings and results for all teams. {len(ratings_dict)} "
                + f"teams with FIFA ratings but {len(self.teams)} teams with results."
            )
        return ratings_dict

    def check_teams_in_ratings(self) -> bool:
        if self.ratings is None:
            return True
        teams = pd.Series(self.teams)
        print('---len(teams[~teams.isin(self.ratings.Team)])')
        if len(teams[~teams.isin(self.ratings.Team)])>0:
            raise ValueError(
                "Must have FIFA ratings and results for all teams. "
                + f"There are {len(teams[~teams.isin(self.ratings.Team)])} teams with no FIFA ratings:\n\n"
                + ', '.join(teams[~teams.isin(self.ratings.Team)]))
        return True

    def set_training_data(self) -> None:
        """Get training data for team model including FIFA ratings as covariates.
        Data returned is for all matches up to specified gameweek and season.
        """
        print("[MODEL FITTING] Setting training data for the model")
        training_data = self.get_result_dict()
        if self.ratings is not None:
            if self.ratings.isna().any().any():
                raise ValueError("There are some NaN values in ratings, please fix or remove")
            if self.check_teams_in_ratings:
                training_data["team_covariates"] = self.get_ratings_dict()
        self.training_data = training_data

    def fit_model(self) -> None:
        """
        Get the team-level stan model, which can give probabilities of
        each potential scoreline in a given fixture.
        """
        if self.training_data is None:
            self.set_training_data()
        print("[MODEL FITTING] Fitting the model")
        self.model = ExtendedDixonColesMatchPredictor().fit(self.training_data)

    def get_fixture_teams(self) -> List[Tuple[str, str]]:
        return [(row.Team_1, row.Team_2) for index,row in self.fixtures.iterrows()]

    def get_fixture_probabilities(self) -> pd.DataFrame:
        """
        Returns probabilities for all fixtures in a given gameweek and season, as a data
        frame with a row for each fixture and columns being home_team,
        away_team, home_win_probability, draw_probability, away_win_probability.
        """
        if self.model is None:
            self.fit_model()
        fixture_teams = self.get_fixture_teams()
        Team_1, Team_2 = zip(*fixture_teams)
        probabilities = self.model.predict_outcome_proba(Team_1, Team_2)
        return pd.DataFrame(
            {
                "Team_1": Team_1,
                "Team_2": Team_2,
                "home_win_probability": probabilities["home_win"],
                "draw_probability": probabilities["draw"],
                "away_win_probability": probabilities["away_win"],
            }
        )

    def get_goal_probabilities_for_fixtures(self, max_goals: int = 10) -> dict[int, dict[str, DeviceArray]]:
        """
        Get the probability that each team in a fixture scores any number of goals up
        to max_goals.
        """
        goals = np.arange(0, max_goals + 1)
        probs = {}
        if self.model is None:
            self.fit_model()
        for index, row in self.fixtures.iterrows():
            home_team_goal_prob = self.model.predict_score_n_proba(
                goals, row.Team_1, row.Team_2, home=False
            )
            away_team_goal_prob = self.model.predict_score_n_proba(
                goals, row.Team_1, row.Team_2, home=False
            )
            probs[index] = {
                row.Team_1: {g: p for g, p in zip(goals, home_team_goal_prob)},
                row.Team_2: {g: p for g, p in zip(goals, away_team_goal_prob)},
            }
        return probs
