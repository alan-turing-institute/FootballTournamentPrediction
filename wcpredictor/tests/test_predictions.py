from wcpredictor import (
    get_teams_data,
    get_results_data,
    get_fifa_rankings_data,
    predict_group_match,
    predict_knockout_match,
    WCPred
)


def test_make_wcpred():
    """
    test that we can create a WCPred object
    """
    results_df = get_results_data()
    wcpred = WCPred(results_df)
    assert isinstance(wcpred, WCPred)


def test_make_wcpred_with_rankings():
    """
    test that we can use a rankings dataframe in WCPred
    """
    results_df = get_results_data()
    rankings_df = get_fifa_rankings_data()
    wcpred = WCPred(results_df, ratings=rankings_df)
    assert isinstance(wcpred, WCPred)
    assert wcpred.ratings is not None


def test_fixture_goal_probs():
    """
    test that we predict more goals for e.g. Brazil than Qatar
    """
    def get_expected_goals(prob_dict, team_name):
        total = 0
        for k, v in prob_dict[team_name].items():
            total += float(k)*float(v)
        return total
    results_df = get_results_data()
    rankings_df = get_fifa_rankings_data()
    wcpred = WCPred(results_df, ratings=rankings_df)
    wcpred.fit_model()
    result = wcpred.get_fixture_goal_probabilities(
        [("Brazil","Qatar")]
    )[0][0]
    assert get_expected_goals(result, "Brazil") > get_expected_goals(result, "Qatar")
    # same for Germany, Australia
    result = wcpred.get_fixture_goal_probabilities(
        [("Australia","Germany")]
    )[0][0]
    assert get_expected_goals(result, "Germany") > get_expected_goals(result, "Australia")
    # same for Portugal, Japan
    result = wcpred.get_fixture_goal_probabilities(
        [("Portugal","Japan")]
    )[0][0]
    assert get_expected_goals(result, "Portugal") > get_expected_goals(result, "Japan")
