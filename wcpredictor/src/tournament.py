"""
Code to run the World Cup tournament, from group stages through the
knockout stages, to the final, and produce a winner.
"""

import random
from time import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .bpl_interface import WCPred
from .data_loader import get_fixture_data, get_results_data, get_teams_data


class Group:
    def __init__(self, name: str, teams: List[str]):
        self.name = name
        self.teams = np.array(teams)

        # "table" is a dictionary keyed by points, goal_difference, goals_for,
        # goals_against, and values arrays with shape (n_teams, n_samples)
        self.table = None

        # "standings" is an array with shape (n_teams, n_samples) with values equal
        # to each teams final group position in each simulation (teams in same order as
        # self.teams)
        self.standings = None

        # results is a dict of played matches
        # dict {
        # "home_team": <team_name>,
        # "away_team": <team_name>,
        # "home_score": <score>,"
        # away_score": <score>"
        # }
        self.results = None

        # order of criteria for deciding group order
        self.metrics = [
            "points",
            "goal_difference",
            "goals_for",
            "head-to-head",
            "random",
        ]

    def calc_table(self) -> None:
        """
        Go through the results, and add points and goals to the table
        """
        home_pts = 3 * (self.results["home_score"] > self.results["away_score"])
        away_pts = 3 * (self.results["home_score"] < self.results["away_score"])
        draw = self.results["home_score"] == self.results["away_score"]
        home_pts[draw] = 1
        away_pts[draw] = 1

        self.table = {
            "points": np.full(
                (len(self.teams), self.results["home_score"].shape[1]), np.nan
            ),
            "goals_for": np.full(
                (len(self.teams), self.results["home_score"].shape[1]), np.nan
            ),
            "goals_against": np.full(
                (len(self.teams), self.results["home_score"].shape[1]), np.nan
            ),
            "goal_difference": np.full(
                (len(self.teams), self.results["home_score"].shape[1]), np.nan
            ),
        }

        for team_idx, team in enumerate(self.teams):
            team_home_idx = self.results["home_team"] == team
            team_home_pts = home_pts[team_home_idx].sum(axis=0)
            team_home_goals_for = self.results["home_score"][team_home_idx].sum(axis=0)
            team_home_goals_against = self.results["away_score"][team_home_idx].sum(
                axis=0
            )

            team_away_idx = self.results["away_team"] == team
            team_away_pts = away_pts[team_away_idx].sum(axis=0)
            team_away_goals_for = self.results["away_score"][team_away_idx].sum(axis=0)
            team_away_goals_against = self.results["home_score"][team_away_idx].sum(
                axis=0
            )

            self.table["points"][team_idx, :] = team_home_pts + team_away_pts
            self.table["goals_for"][team_idx, :] = (
                team_home_goals_for + team_away_goals_for
            )
            self.table["goals_against"][team_idx, :] = (
                team_home_goals_against + team_away_goals_against
            )
            self.table["goal_difference"][team_idx, :] = (
                self.table["goals_for"][team_idx, :]
                - self.table["goals_against"][team_idx, :]
            )

    def get_qualifiers(self) -> Tuple:
        """
        return the two teams that topped the group
        """
        if self.standings is None:
            self.calc_standings()
        first = np.nonzero(self.standings.T == 1)[1]
        second = np.nonzero(self.standings.T == 2)[1]
        return self.teams[first], self.teams[second]

    def fill_standings_position(
        self, sample, team_idx: int, position: int, verbose: bool = False
    ) -> None:
        """
        Fill specified slot in our team standings.
        """
        if verbose:
            print(f"Putting {self.teams[team_idx]} in {position}")
        self.standings[team_idx, sample] = position
        return

    def find_head_to_head_winner(
        self, sample: int, team_A: int, team_B: int
    ) -> Tuple[int, int]:
        fixture_id = np.where(
            (
                (self.results["home_team"] == self.teams[team_A])
                | (self.results["away_team"] == self.teams[team_A])
            )
            & (
                (self.results["home_team"] == self.teams[team_B])
                | (self.results["away_team"] == self.teams[team_B])
            )
        )[0]
        if len(fixture_id) != 1:
            raise KeyError(
                f"No unique fixture between {self.teams[team_A]} and "
                f"{self.teams[team_B]}"
            )
        if self.results["home_team"][fixture_id] == self.teams[team_A]:
            # team_A is the home team
            if (
                self.results["home_score"][fixture_id, sample]
                > self.results["away_score"][fixture_id, sample]
            ):
                return team_A, team_B
            if (
                self.results["home_score"][fixture_id, sample]
                < self.results["away_score"][fixture_id, sample]
            ):
                return team_B, team_A
        else:
            # team_B is the home team
            if (
                self.results["home_score"][fixture_id, sample]
                > self.results["away_score"][fixture_id, sample]
            ):
                return team_B, team_A
            if (
                self.results["home_score"][fixture_id, sample]
                < self.results["away_score"][fixture_id, sample]
            ):
                return team_A, team_B

        return None, None

    def set_positions_using_metric(
        self,
        sample,
        teams_to_sort: List[int],
        positions_to_fill: List[int],
        metric: str,
        verbose: bool = False,
    ) -> None:
        """Recursively called on subsequent metrics until all the positions in the table
        can be filled without ties (with ties broken at random if all other metrics are
        tied)"""
        if len(teams_to_sort) != len(positions_to_fill):
            raise RuntimeError(
                f"Can't fill {len(positions_to_fill)} positions with "
                f"{len(teams_to_sort)} teams"
            )
        if verbose:
            print(
                f"Sorting {self.teams[teams_to_sort]} using {metric} to fill positions "
                f"{positions_to_fill}"
            )

        if metric == "head-to-head":
            if len(teams_to_sort) > 2:
                if verbose:
                    print("Can't use head-to-head for more than 2 teams")
                # skip ahead to random
                self.set_positions_using_metric(
                    sample, teams_to_sort, positions_to_fill, "random"
                )
            else:
                team_1, team_2 = self.find_head_to_head_winner(
                    sample, teams_to_sort[0], teams_to_sort[1]
                )
                if (
                    team_1 is not None and team_2 is not None
                ):  # not null if there was a winner
                    self.fill_standings_position(sample, team_1, positions_to_fill[0])
                    self.fill_standings_position(sample, team_2, positions_to_fill[1])
                else:
                    # go to random
                    self.set_positions_using_metric(
                        sample, teams_to_sort, positions_to_fill, "random"
                    )
            return
        elif metric == "random":
            # if random, just shuffle our list
            random.shuffle(teams_to_sort)
            for i, pos in enumerate(positions_to_fill):
                self.fill_standings_position(sample, teams_to_sort[i], pos)
            if verbose:
                print(f"randomly assigned {len(teams_to_sort)} teams")
            return

        # ok, otherwise we need to sort the table by the metric (minus sign to sort
        # descending)
        team_list = teams_to_sort[
            np.argsort(-self.table[metric][teams_to_sort, sample])
        ]
        team_scores = self.table[metric][team_list, sample]

        if verbose:
            print("team order", team_list)
            print(metric, team_scores)

        # figure out the next metric, in case this one doesn't differentiate
        current_metric_index = self.metrics.index(metric)
        new_metric = self.metrics[current_metric_index + 1]

        # OK, let's get sorting!! Start with two-team case
        if len(team_list) == 2:
            if team_scores[0] > team_scores[1]:  # one team is better
                self.fill_standings_position(sample, team_list[0], positions_to_fill[0])
                self.fill_standings_position(sample, team_list[1], positions_to_fill[1])
            else:
                # they are equal - call this func again with the next metric
                self.set_positions_using_metric(
                    sample, team_list, positions_to_fill, new_metric
                )
            return
        elif len(team_list) == 3:
            # 4 possible cases
            if team_scores[0] > team_scores[1] > team_scores[2]:  # 1st > 2nd > 3rd
                self.fill_standings_position(sample, team_list[0], positions_to_fill[0])
                self.fill_standings_position(sample, team_list[1], positions_to_fill[1])
                self.fill_standings_position(sample, team_list[2], positions_to_fill[2])
                return  # we are done!
            elif (
                team_scores[0] > team_scores[1] and team_scores[1] == team_scores[2]
            ):  # last two equal
                self.fill_standings_position(sample, team_list[0], positions_to_fill[0])
                # call this func again with the last two, and the next metric
                self.set_positions_using_metric(
                    sample, team_list[1:], positions_to_fill[1:], new_metric
                )
                return
            elif (
                team_scores[0] == team_scores[1] and team_scores[1] > team_scores[2]
            ):  # first two equal
                self.fill_standings_position(sample, team_list[2], positions_to_fill[2])
                # call this func again with the first two, and the next metric
                self.set_positions_using_metric(
                    sample, team_list[:2], positions_to_fill[:2], new_metric
                )
            else:  # all three teams equal - just move onto the next metric
                self.set_positions_using_metric(
                    sample, team_list, positions_to_fill, new_metric
                )
            return
        elif len(team_list) == 4:  # 8 possible cases.
            if (
                team_scores[0] > team_scores[1]
                and team_scores[1] > team_scores[2]
                and team_scores[2] > team_scores[3]
            ):  # case 1) all in order
                self.fill_standings_position(sample, team_list[0], 1)
                self.fill_standings_position(sample, team_list[1], 2)
                self.fill_standings_position(sample, team_list[2], 3)
                self.fill_standings_position(sample, team_list[3], 4)
                # we are done!
                return
            elif (
                team_scores[0] == team_scores[1]
                and team_scores[1] > team_scores[2]
                and team_scores[2] > team_scores[3]
            ):  # case 2) first two equal
                self.fill_standings_position(sample, team_list[2], 3)
                self.fill_standings_position(sample, team_list[3], 4)
                # call this func with the first two and the next metric
                self.set_positions_using_metric(
                    sample, team_list[:2], positions_to_fill[:2], new_metric
                )
            elif (
                team_scores[0] > team_scores[1]
                and team_scores[1] == team_scores[2]
                and team_scores[2] > team_scores[3]
            ):  # case 3) middle two equal
                self.fill_standings_position(sample, team_list[0], 1)
                self.fill_standings_position(sample, team_list[3], 4)
                # call this func with the middle two and the next metric
                self.set_positions_using_metric(
                    sample, team_list[1:3], positions_to_fill[1:3], new_metric
                )
            elif (
                team_scores[0] > team_scores[1]
                and team_scores[1] > team_scores[2]
                and team_scores[2] == team_scores[3]
            ):  # case 4) last two equal
                self.fill_standings_position(sample, team_list[0], 1)
                self.fill_standings_position(sample, team_list[1], 2)
                # call this func with the last two and the next metric
                self.set_positions_using_metric(
                    sample, team_list[2:], positions_to_fill[2:], new_metric
                )
            elif (
                team_scores[0] == team_scores[1]
                and team_scores[1] == team_scores[2]
                and team_scores[2] > team_scores[3]
            ):  # case 5) all equal except last
                self.fill_standings_position(sample, team_list[3], 4)
                # call this func with the first three and the next metric
                self.set_positions_using_metric(
                    sample, team_list[:3], positions_to_fill[:3], new_metric
                )
            elif (
                team_scores[0] > team_scores[1]
                and team_scores[1] == team_scores[2]
                and team_scores[2] == team_scores[3]
            ):  # case 6) all equal except first
                self.fill_standings_position(sample, team_list[0], 1)
                # call this func with the last three and the next metric
                self.set_positions_using_metric(
                    sample, team_list[1:], positions_to_fill[1:], new_metric
                )
            elif (
                team_scores[0] == team_scores[1]
                and team_scores[1] > team_scores[2]
                and team_scores[2] == team_scores[3]
            ):  # case 7) nightmare scenario!!
                # call func with first two and next metric
                self.set_positions_using_metric(
                    sample, team_list[:2], positions_to_fill[:2], new_metric
                )
                # call func with last two and next metric
                self.set_positions_using_metric(
                    sample, team_list[2:], positions_to_fill[2:], new_metric
                )
            else:  # case 8) all equal - carry on to next metric
                # call this func with the last three and the next metric
                self.set_positions_using_metric(
                    sample, team_list, positions_to_fill, new_metric
                )

    def calc_standings(self, head_to_head=True, verbose=False) -> None:
        """
        sort the table, and try and assign positions in the standings

        if not head_to_head sort by points -> goal difference -> goals -> random
        (i.e. don't consider head to head as a tiebreaker)
        """
        if self.table is None:
            self.calc_table()

        if not head_to_head:
            self.standings = len(self.teams) - np.lexsort(
                (
                    np.random.random(size=self.table["points"].shape),
                    self.table["goals_for"],
                    self.table["goal_difference"],
                    self.table["points"],
                ),
                axis=0,
            ).argsort(axis=0)
        else:
            # Include head_to_head tiebreaker between two teams (slower)
            # reset the standings table to start from scratch
            self.standings = np.empty(self.table["points"].shape, dtype=int)
            # now calculate the standings again
            for sample in range(self.table["points"].shape[1]):
                if verbose:
                    print("sample", sample)
                self.set_positions_using_metric(
                    sample, np.arange(len(self.teams)), [1, 2, 3, 4], "points"
                )
                if verbose:
                    print("-" * 15)

    def add_results(self, results):
        """
        Add a results for group-stage matches.
        Parameters
        ==========
        results: Simulated match scores with keys home_team, away_team, home_score,
        away_score. These will be filtered to only include fixtures that are between
        teams in the group.
        """
        group_mask = np.isin(results["home_team"], self.teams)
        self.results = {
            "home_team": results["home_team"][group_mask],
            "away_team": results["away_team"][group_mask],
            "home_score": np.array(results["home_score"])[group_mask],
            "away_score": np.array(results["away_score"])[group_mask],
        }

    def __str__(self) -> str:
        max_team_name_length = max(len(name) for name in self.teams)
        output = f"Position | Team{' '*(max_team_name_length-3)}| Points | GF   | GA \n"
        self.calc_standings()
        mean_rank = self.standings.mean(axis=1)
        mean_pts = self.table["points"].mean(axis=1)
        mean_gf = self.table["goals_for"].mean(axis=1)
        mean_ga = self.table["goals_against"].mean(axis=1)
        for idx in np.argsort(mean_rank):
            output += (
                f"    {mean_rank[idx]:.2f}   {self.teams[idx]}"
                f"{' '*(max_team_name_length-len(self.teams[idx]))} "
                f"  {mean_pts[idx]:.2f}     {mean_gf[idx]:.2f}   {mean_ga[idx]:.2f} \n"
            )
        return output


class Tournament:
    def __init__(
        self,
        year: str = "2022",
        num_samples: int = 1,
        resume_from: Optional[str] = None,
    ):
        self.year = year
        self.teams_df = get_teams_data(year)
        self.fixtures_df = get_fixture_data(year)
        self.group_names = list(set(self.teams_df["Group"].values))
        self.groups = {}
        for n in self.group_names:
            g = Group(n, list(self.teams_df[self.teams_df["Group"] == n].Team.values))
            self.groups[n] = g
        self.bracket = pd.DataFrame(index=np.arange(num_samples))
        self.is_complete = False
        self.num_samples = num_samples
        self.stage_counts = None
        if resume_from is not None:
            if resume_from in []:
                # end date from tournament round fixture dates
                ...
            r, _ = get_results_data(
                start_date=f"{year}-01-01", end_date=resume_from, competitions="W"
            )
            self.actual_results = ...

    def play_group_stage(
        self,
        wc_pred: WCPred,
        seed: Optional[int] = None,
        head_to_head: bool = True,
    ) -> None:
        print("Group")
        t = time()
        group_fixtures = self.fixtures_df[self.fixtures_df.Stage == "Group"]
        results = wc_pred.sample_score(
            group_fixtures["Team_1"],
            group_fixtures["Team_2"],
            seed=seed,
            num_samples=self.num_samples,
        )
        for g in self.groups.values():
            g.add_results(results)
            g.calc_standings(head_to_head=head_to_head)
        print(time() - t)

    def play_knockout_stages(
        self, wc_pred: WCPred, seed: Optional[int] = None, verbose: bool = False
    ) -> None:
        """
        For the round of 16, assign the first and second place teams
        from each group to the aliases e.g. "A1", "B2"
        """
        for g in self.groups.values():
            t1, t2 = g.get_qualifiers()
            self.bracket["1" + g.name] = t1
            self.bracket["2" + g.name] = t2

        for stage in ["R16", "QF", "SF", "F"]:
            print(stage)
            t = time()
            stage_fixtures = self.fixtures_df[self.fixtures_df["Stage"] == stage]

            results = wc_pred.sample_outcome(
                self.bracket[stage_fixtures["Team_1"]].values.flatten(),
                self.bracket[stage_fixtures["Team_2"]].values.flatten(),
                knockout=True,
                seed=seed,
                num_samples=1,
            ).reshape((self.num_samples, len(stage_fixtures)))

            self.bracket[stage_fixtures["Team_1"] + stage_fixtures["Team_2"]] = results

            if stage == "F":
                self.winner = results.flatten()
            print(time() - t)

        self.is_complete = True

    def count_stages(self):
        """
        Count how far teams got in the tournament.
        """
        if not self.is_complete:
            raise RuntimeError("Tournament is not yet complete")

        stages = ["F", "SF", "QF", "R16", "Group"]
        self.stage_counts = pd.DataFrame(columns=stages, index=self.teams_df["Team"])
        self.stage_counts["Group"] = self.num_samples
        for stage in stages[:-1]:
            round_aliases = np.unique(
                self.fixtures_df.loc[
                    self.fixtures_df.Stage == stage, ["Team_1", "Team_2"]
                ]
            )
            teams, counts = np.unique(self.bracket[round_aliases], return_counts=True)
            self.stage_counts[stage] = pd.Series(counts, teams)

        # 0 counts will appear as NaN, replace them
        self.stage_counts = self.stage_counts.fillna(0)
        # exact round knocked out at (rather than cummulative progression)
        self.stage_counts = self.stage_counts.diff(axis=1)
        self.stage_counts["F"] = self.num_samples - self.stage_counts.sum(axis=1)

        self.stage_counts["W"] = pd.Series(self.winner).value_counts()
        self.stage_counts["W"] = self.stage_counts["W"].fillna(0)
        self.stage_counts["RU"] = self.stage_counts["F"] - self.stage_counts["W"]
        self.stage_counts = self.stage_counts[
            ["Group", "R16", "QF", "SF", "RU", "W"]
        ].astype(int)
