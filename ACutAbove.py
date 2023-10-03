import streamlit as st
import pandas as pd
from sleeper_wrapper import League

leagueid = "992211821861576704"

league = League(leagueid)

league = League(leagueid)
rosters = league.get_rosters()
users = league.get_users()
standings = league.get_standings(rosters,users)

st.write(standings)
st.write(pd.DataFrame(standings))
