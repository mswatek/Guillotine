import streamlit as st
import pandas as pd
from sleeper.api import LeagueAPIClient
from sleeper.enum import Sport
from sleeper.model import (
    League,
    Roster,
    User,
    Matchup,
    PlayoffMatchup,
    Transaction,
    TradedPick,
    SportState,
)

####### setting some stuff up

st.title(":blue[A Cut Above]")

leagueid = "992211821861576704"



if __name__ == "__main__":
    # get a league by its ID
    league: League = LeagueAPIClient.get_league(league_id=leagueid)

    # get all rosters in a particular league
    league_rosters: list[Roster] = LeagueAPIClient.get_rosters(league_id=leagueid)

    rosters = pd.DataFrame(league_rosters)
    rosters[['division','fpts', 'fpts_against','fpts_against_decimal','fpts_decimal','losses','ppts','ppts_decimal','ties','total_moves','waiver_adjusted','waiver_budget_used','waiver_position','wins']] = rosters['settings'].apply(pd.Series)

    st.write(rosters)
