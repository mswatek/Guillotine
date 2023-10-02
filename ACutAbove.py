import streamlit as st
import requests
import pandas as pd
import json
import functools as ft
from time import strftime, localtime
from datetime import datetime
import numpy as np
import altair as alt
#import IPython
import plotly.express as px
from pandas import json_normalize
import seaborn as sns
from sleeper.api import PlayerAPIClient
from sleeper.enum import Sport, TrendType
from sleeper.model import Player, PlayerTrend
from sleeper.api import LeagueAPIClient
from sleeper.api import UserAPIClient
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overall", "Waivers","Players", "Managers", "TABLES!"])

now = datetime.now()
now = now.strftime('%Y-%m-%d')



if  now > '2024-01-01': currentweek='18'
elif now > '2023-12-25': currentweek='17'
elif now > '2023-12-18': currentweek='16'
elif now > '2023-12-11': currentweek='15'
elif now > '2023-12-04': currentweek='14'
elif now > '2023-11-27': currentweek='13'
elif now > '2023-11-20': currentweek='12'
elif now > '2023-11-13': currentweek='11'
elif now > '2023-11-06': currentweek='10'
elif now > '2023-10-30': currentweek='9'
elif now > '2023-10-23': currentweek='8'
elif now > '2023-10-16': currentweek='7'
elif now > '2023-10-09': currentweek='6'
elif now > '2023-10-02': currentweek='5'
elif now > '2023-09-25': currentweek=4
elif now > '2023-09-18': currentweek='3'
elif now > '2023-09-11': currentweek='2'
else: currentweek='1'


st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 2px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}

</style>""", unsafe_allow_html=True)

############################################# users

# get all users in a particular league
league_users: list[User] = LeagueAPIClient.get_users_in_league(league_id=leagueid)

users = pd.DataFrame(league_users)
userlist = users['user_id'].tolist() ## metadata has team name...maybe I can eventually get this to be fully automated

# initialize list of lists
data = [['Mat', 1], ['Jeff', 2], ['CJ', 3],['Leo', 4],['Kevin', 5],['Hunter', 6],['Kyle', 7], ['Nick', 8], \
['Jimmy', 9],['Jonathan', 10],['Jon', 11],['Harry J', 12],['Ian', 13],['Brandon', 14],['Myles', 15],['Harry G', 16],['Shea', 17],['Ed', 18]]
  
# Create the pandas DataFrame
users_df = pd.DataFrame(data, columns=['Manager', 'roster_id'])


if __name__ == "__main__":
    # get a league by its ID
    league: League = LeagueAPIClient.get_league(league_id="992211821861576704")



############################################# rosters

# get all rosters in a particular league
league_rosters: list[Roster] = LeagueAPIClient.get_rosters(league_id="992211821861576704")

rosters = pd.DataFrame(league_rosters)
rosters[['division','fpts', 'fpts_against','fpts_against_decimal','fpts_decimal','losses','ppts','ppts_decimal','ties','total_moves','waiver_adjusted','waiver_budget_used','waiver_position','wins']] = rosters['settings'].apply(pd.Series)


rosters = pd.merge(rosters, users_df, left_on='roster_id', right_on='roster_id')
rosters = rosters[['Manager', 'fpts','fpts_decimal','ppts','ppts_decimal','waiver_budget_used']]

############################################# players

if __name__ == "__main__":
    # get all players in a particular sport
    nfl_players: dict[str, Player] = PlayerAPIClient.get_all_players(sport=Sport.NFL)

    # Use pandas.DataFrame.from_dict() to Convert JSON to DataFrame
    df2 = pd.DataFrame.from_dict(nfl_players, orient="index")

    test = df2[["first_name","last_name","position","team","player_id"]].reset_index()
    test.rename(columns={'position': 'Position','team':'Team'},inplace=True)

    test['Position'] = test['Position'].astype('string').str.replace('NFLPosition.', '')
    test['Team'] = test['Team'].astype('string').str.replace('NFLTeam.', '')



############################################# matchups

# get all matchups in a week for a particular league
    all_matchups: list[Matchup] = LeagueAPIClient.get_matchups_for_week(
        league_id="992211821861576704", week=1
    )

all_matchups1=pd.DataFrame()
for i in range(1,currentweek+1): #gotta automate this
    data: list[Matchup] = LeagueAPIClient.get_matchups_for_week(league_id=leagueid, week=i)
    data1 = pd.DataFrame(data)
    data1['Week'] = i
    frames = [all_matchups1,data1]
    all_matchups1= pd.concat(frames)


all_matchups = pd.merge(all_matchups1, users_df, left_on='roster_id', right_on='roster_id')

all_matchups['players_points'] = all_matchups['players_points'].astype('string')
all_matchups['Status'] = np.where((all_matchups['players_points']=='{}'), "Out", "Alive")
all_matchups['Points'] = np.where((all_matchups['points'] == 0) & (all_matchups['Status']=='Out'), [None], all_matchups['points'])
all_matchups = all_matchups[["Week","Manager","Points","Status"]]
all_matchups['Week'] = all_matchups['Week'].astype('string')
all_matchups['Points'] = all_matchups['Points'].astype('string').astype('float')
all_matchups['Cumulative Points'] = all_matchups.groupby(['Manager'])['Points'].cumsum()
all_matchups['2-Week Rolling Avg'] = all_matchups.groupby('Manager')['Points'].transform(lambda x: x.rolling(2, 2).mean())
all_matchups['3-Week Rolling Avg'] = all_matchups.groupby('Manager')['Points'].transform(lambda x: x.rolling(3, 3).mean())
all_matchups['3-Week Rolling Avg'] = np.where(all_matchups['Week'] == '1',all_matchups['Points'], ##bring in week 1 and rolling week 2 to make full graph
        np.where(all_matchups['Week']=='2',all_matchups['2-Week Rolling Avg'],all_matchups['3-Week Rolling Avg']))
all_matchups["Rolling Rank"] = all_matchups.groupby("Week")["3-Week Rolling Avg"].rank(method="dense", ascending=False)

weekly_points = px.line(all_matchups, x="Week", y="Points", color='Manager',title="Points by Week")
weekly_points_cumu = px.line(all_matchups, x="Week", y="Cumulative Points", color='Manager',title="Cumulative Points by Week")
weekly_dist = px.box(all_matchups, x="Week", y="Points",points="all",title="Weekly Distribution") #px.strip take out boxes


#####waterfall table

order_df = all_matchups
order_df['outorder'] = np.where(order_df['Status']=='Out',1,0)
order_df['outorder2'] = order_df.groupby(['Manager'])['outorder'].cumsum()
order_df = order_df.drop_duplicates(subset=['Manager'], keep='last')
order_df = order_df.sort_values(by = ['outorder2', 'Cumulative Points'], ascending = [False, True], na_position = 'first')

manager_order = order_df['Manager'].tolist()

all_matchups_wide = all_matchups[["Week","Manager","Points"]]
all_matchups_wide = all_matchups_wide.pivot(index='Week', columns='Manager', values='Points')

all_matchups_wide = all_matchups_wide.reindex(columns=manager_order)

##waterfall color palette options
cm = sns.color_palette("blend:red,yellow,green", as_cmap=True) # option 1
def color_survived(val): # option 2
    color = 'red' if val<50 else 'yellow' if val<75 else 'green' if val>74 else 'white'
    return f'background-color: {color}'

############################################# transactions
    
all_trans=pd.DataFrame()
for i in range(1,5): #gotta automate this
    data: list[Transaction] = LeagueAPIClient.get_transactions(league_id=leagueid, week=i)
    data1 = pd.DataFrame(data)
    frames = [all_trans,data1]
    all_trans= pd.concat(frames)

result = all_trans
result  = result.explode('adds')
result  = result.explode('drops')
result[['seq','waiver_bid']] = result['settings'].apply(pd.Series)
result[['notes']] = result['metadata'].apply(pd.Series)
result['roster_ids'] = result['roster_ids'].astype('string')
result['roster_ids'] = result['roster_ids'] .apply(lambda x: x.strip('[]'))
result['roster_ids'] = result['roster_ids'].astype('int64')
result = pd.merge(result, users_df, left_on='roster_ids', right_on='roster_id')
result['type'] = result['type'].astype('string')
result['type'] = result['type'].replace(["TransactionType.WAIVER", "TransactionType.FREE_AGENT",""], ["Waiver","Free Agent","Commissioner"])
result['type'] = result['type'].fillna("Commissioner")

result['status'] = result['status'].astype('string')
result['status'] = result['status'].replace(["TransactionStatus.COMPLETE", "TransactionStatus.FAILED"], ["Complete","Failed"])


result['status_updated'] = result['status_updated'].astype('string')
result['status_updated'] = result['status_updated'].str.replace(',', '')
result['status_updated'] = result['status_updated'].astype(float)
result['status_new'] = result['status_updated'].div(1000)

result['date2'] = pd.to_datetime((result['status_new']),unit='s',utc=True).map(lambda x: x.tz_convert('US/Pacific'))
result['date3'] = result['date2'].dt.date
result['day_of_week'] = result['date2'].dt.day_name()

conditions = [
    (result['date2'] < '2023-09-11'),
    (result['date2'] > '2023-10-23'),
    (result['date2'] > '2023-10-16'),
    (result['date2'] > '2023-10-09'),
    (result['date2'] > '2023-10-02'),
    (result['date2'] > '2023-09-25'),
    (result['date2'] > '2023-09-18'),
    (result['date2'] > '2023-09-11')
    ]

values = ['1','8','7','6','5','4','3','2']

result['week'] = np.select(conditions, values) #as defined at the top

added_df = pd.merge(result, test, left_on='adds', right_on='player_id')
added_df['Name'] = added_df['first_name'] + ' ' + added_df['last_name']

############ cumulative transactions

transactions_df = added_df[['week','Manager','Name','Position','Team','type','status','waiver_bid','notes']].query("status == 'Complete'")

week_manager_df = transactions_df.query("type == 'Waiver'").groupby(['week','Manager']).agg(WinningBids=('waiver_bid', 'count'),MoneySpent=('waiver_bid', 'sum'),MaxPlayer=('waiver_bid', 'max'),MedianPlayer=('waiver_bid', 'median')).reset_index()
week_manager_df = pd.merge(all_matchups, week_manager_df,left_on=['Week','Manager'], right_on=['week','Manager'],how='left')
week_manager_df['MoneySpent'] = np.where((week_manager_df['MoneySpent'].isnull()), 0, week_manager_df['MoneySpent'])
week_manager_df['Cumulative Spend'] = week_manager_df.groupby(['Manager'])['MoneySpent'].cumsum()
week_manager_df['Remaining Budget'] = np.where((week_manager_df['Status']=='Out'), 0, (1000-week_manager_df['Cumulative Spend']))

manager_position_df = transactions_df.query("type == 'Waiver'").groupby(['Position','Manager']).agg(WinningBids=('waiver_bid', 'count'),MoneySpent=('waiver_bid', 'sum'),MaxPlayer=('waiver_bid', 'max'),MedianPlayer=('waiver_bid', 'median')).reset_index()
week_position_df = transactions_df.query("type == 'Waiver'").groupby(['week','Position']).agg(WinningBids=('waiver_bid', 'count'),MoneySpent=('waiver_bid', 'sum'),MaxPlayer=('waiver_bid', 'max'),MedianPlayer=('waiver_bid', 'median')).reset_index()
week_budget_df = week_manager_df.groupby(['Week']).agg(RemainingBudget=('Remaining Budget', 'sum')).reset_index()


week_overall_df = transactions_df.query("type == 'Waiver'").groupby(['week']).agg(WinningBids=('waiver_bid', 'count'),MoneySpent=('waiver_bid', 'sum'),MaxPlayer=('waiver_bid', 'max'),MedianPlayer=('waiver_bid', 'median')).reset_index()
position_overall_df = transactions_df.query("type == 'Waiver'").groupby(['Position']).agg(WinningBids=('waiver_bid', 'count'),MoneySpent=('waiver_bid', 'sum'),MaxPlayer=('waiver_bid', 'max'),MedianPlayer=('waiver_bid', 'median')).reset_index()
manager_overall_df = transactions_df.query("type == 'Waiver'").groupby(['Manager']).agg(WinningBids=('waiver_bid', 'count'),MoneySpent=('waiver_bid', 'sum'),MaxPlayer=('waiver_bid', 'max'),MedianPlayer=('waiver_bid', 'median')).reset_index()


###budget charts
week_manager_budget = px.line(week_manager_df, x="Week", y="Remaining Budget", color='Manager',markers=True,title="Budget Remaining by Week")
week_budget_chart = px.bar(week_budget_df, x="Week", y="RemainingBudget",text_auto='.2s',title="Overall Budget Remaining by Week")

############# adds


adds_df = added_df.groupby(['Manager','Name','Position','Team','type','status','adds','week','waiver_bid','notes'])['leg'].count().reset_index(name="Count")

manager_player_success = adds_df.query("status == 'Complete'").groupby(['Manager','Name','Position','Team','type','week'])['waiver_bid'].max().reset_index(name="Winning Bid")
manager_player_fail = adds_df.query("notes == 'This player was claimed by another owner.'").groupby(['Manager','Name','Position','Team','type','week'])['waiver_bid'].count().reset_index(name="Losing Bids")
manager_player_fail_max = adds_df.query("notes == 'This player was claimed by another owner.'").groupby(['Manager','Name','Position','Team','type','week'])['waiver_bid'].max().reset_index(name="Max Losing Bid")
manager_player_roster = adds_df.groupby(['Manager','Name','Position','Team','type','week'])['notes'].apply(lambda x: (x=='Unfortunately, your roster will have too many players after this transaction.').sum()).reset_index(name='No Space Count')

adds_dataframes = [manager_player_success,manager_player_fail, manager_player_fail_max,manager_player_roster]

adds_df_combined = ft.reduce(lambda left, right: pd.merge(left, right,on=['Manager', 'Name','Position','Team','type','week'],how='outer'), adds_dataframes)
adds_df_combined['Losing Bids'] = np.where((adds_df_combined['Winning Bid'] > 0), [None], adds_df_combined['Losing Bids'])
adds_df_combined['Max Losing Bid'] = np.where((adds_df_combined['Winning Bid'] > 0), [None], adds_df_combined['Max Losing Bid'])


adds_player = adds_df_combined.query("type == 'Waiver'").groupby(['week','Name','Position','Team']) \
    .agg(Bids=('Position', 'count'),WinningBid=('Winning Bid', 'max'),LosingBids=('Losing Bids', 'count'), LosingAmounts=('Max Losing Bid', 'sum'),\
         LosingMax=('Max Losing Bid', 'max'),LosingMin=('Max Losing Bid', 'min'),LosingAvg=('Max Losing Bid', 'mean'),LosingMedian=('Max Losing Bid', 'median')).reset_index()

adds_player = adds_player.sort_values(by='WinningBid',ascending=False)
adds_player['Difference'] = adds_player['WinningBid'] - adds_player['LosingMax']

##bring in manager for top and second-highest bid
adds_manager = adds_df[['Name','Manager','week','waiver_bid']]

adds_player = pd.merge(adds_player, adds_manager, left_on=['week','Name','WinningBid'], right_on=['week','Name','waiver_bid'],how='left').drop_duplicates().reset_index()
adds_player = pd.merge(adds_player, adds_manager, left_on=['week','Name','LosingMax'], right_on=['week','Name','waiver_bid'],how='left').drop_duplicates().reset_index()
adds_player['check_dupes'] = adds_player.duplicated(subset=['week','Name','Manager_x','waiver_bid_x'], keep=False).astype(int).astype(float)

##dedupe for bids where multiple people had second highest
adds_duped = np.where(adds_player['check_dupes']==1)
adds_duped = adds_player.query("check_dupes==1")
adds_duped['CUM_CONCAT']=[y.Manager_y.tolist()[:z+1] for x, y in adds_duped.groupby(['week','Name','Manager_x','waiver_bid_x'])for z in range(len(y))]
adds_duped['CUM_CONCAT'] = adds_duped['CUM_CONCAT'].astype('string')
adds_duped['cum_bids'] = adds_duped.groupby(['week','Name','Manager_x','waiver_bid_x'])['check_dupes'].cumsum()
adds_duped = adds_duped.drop_duplicates(subset=['week','Name','Manager_x','waiver_bid_x'], keep='last')
adds_duped = adds_duped[['week','Name','Manager_x','CUM_CONCAT']]

adds_player = pd.merge(adds_player, adds_duped, left_on=['week','Name','Manager_x'], right_on=['week','Name','Manager_x'],how='left')
adds_player['Manager_y'] = np.where((adds_player['check_dupes']==1), adds_player['CUM_CONCAT'], adds_player['Manager_y'])
adds_player = adds_player.drop_duplicates(subset=['week','Name','Manager_x','waiver_bid_x'], keep='last')
adds_player = adds_player.drop(['waiver_bid_x', 'waiver_bid_y','check_dupes','CUM_CONCAT'], axis=1)
adds_player.rename(columns={'Manager_x': 'Winning Manager','Manager_y':'Runner-Up Manager'},inplace=True)

bids_winning = adds_player.groupby(['Winning Manager']) \
    .agg(AvgWinGap=('Difference', 'mean'),MedianWinGap=('Difference', 'median'),\
         MaxWinGap=('Difference', 'max'),MinWinGap=('Difference', 'min')).reset_index()

bids_runnerup = adds_player.groupby(['Runner-Up Manager']) \
    .agg(AvgGap=('Difference', 'mean'),MedianGap=('Difference', 'median'),\
         MaxGap=('Difference', 'max'),MinGap=('Difference', 'min')).reset_index()


adds_player_summary = adds_player.groupby(['Name','Position','Team']) \
    .agg(Pickups=('Bids','count'),AvgBids=('Bids','mean'),MoneySpent=('WinningBid','sum'),AvgSpent=('WinningBid','mean')).reset_index()

waiver_scatter = px.scatter(adds_player, x="WinningBid", y="Difference",size="Bids", color="Bids",hover_data="Name",title="Waiver Bids")\
    .add_shape(type="line",x0=4, y0=0, x1=max(adds_player['WinningBid']), y1=max(adds_player['Difference']),line=dict(color="MediumPurple",width=4,dash="dot"))

##player tree map
player_tree = px.treemap(adds_player, path=['Position', 'Name'], values='WinningBid',
                  color='Position', hover_data=['Name'],title="Tree Map of Waivers by Position")

############# drops
dropped_df = pd.merge(result, test, left_on='drops', right_on='player_id')
dropped_df['Name'] = dropped_df['first_name'] + ' ' + dropped_df['last_name']
eliminated_df = dropped_df.loc[dropped_df['type'] == 'Commissioner']
released_df = dropped_df.loc[(dropped_df['type'].isin(['Waiver','Free Agent'])) & (dropped_df['status'] =='Complete')]

eliminated_summary = eliminated_df.groupby(['Name','Position','Team'])['leg'].count().sort_values(ascending=False).reset_index(name='Times Eliminated')
released_summary = released_df.groupby(['Name','Position','Team'])['leg'].count().sort_values(ascending=False).reset_index(name='Times Dropped')


#####power rankings

power_rankings = week_manager_df.loc[(week_manager_df['Week']== week_manager_df['Week'].max()) & (week_manager_df['Status']=='Alive')]
power_rankings['rolling_z'] = (power_rankings['3-Week Rolling Avg'] - power_rankings['3-Week Rolling Avg'].mean())/power_rankings['3-Week Rolling Avg'].std(ddof=0)
power_rankings['budget_z'] = (power_rankings['Remaining Budget'] - power_rankings['Remaining Budget'].mean())/power_rankings['Remaining Budget'].std(ddof=0)
power_rankings['Power Ranking'] = power_rankings['rolling_z'] + power_rankings['budget_z']
#power_rankings["Power Ranking"] = power_rankings["Rank Sum"].rank(method="dense", ascending=False) #maybe keep this out for now?

power_rankings = power_rankings[['Manager','3-Week Rolling Avg','Remaining Budget','Power Ranking']].sort_values(by = 'Power Ranking',ascending=False)

#color palette options
cm_power = sns.light_palette("green", as_cmap=True)


################ add automated text here

##tab1
budget_left_text = min(week_budget_df['RemainingBudget'])
alive_text = 19-currentweek

lost_teams_text = week_manager_df.loc[(week_manager_df['Status'] == 'Out') & \
                                       (week_manager_df['Week'] == week_manager_df['Week'].max()), 'Manager']
lost_teams_text = list(lost_teams_text)
lost_teams_text = " and ".join([str(item) for item in lost_teams_text])

most_points_text = all_matchups.loc[all_matchups['Cumulative Points'] == all_matchups['Cumulative Points'].max(), 'Manager'].values[0]
most_rpoints_text = all_matchups.loc[(all_matchups['Rolling Rank'] == all_matchups['Rolling Rank'].min()) & \
                                     (all_matchups['Week'] == all_matchups['Week'].max()),'Manager'].values[0]

pr_top = power_rankings.loc[power_rankings['Power Ranking'] == power_rankings['Power Ranking'].max(), 'Manager'].values[0]
pr_bottom = power_rankings.loc[power_rankings['Power Ranking'] == power_rankings['Power Ranking'].min(), 'Manager'].values[0]


##tab2
most_budget_text = week_manager_df.loc[(week_manager_df['Remaining Budget'] == week_manager_df['Remaining Budget'].max()) & \
                                       (week_manager_df['Week'] == week_manager_df['Week'].max()), 'Manager'].values[0]

max_position_text = position_overall_df.loc[position_overall_df['MoneySpent'] == position_overall_df['MoneySpent'].max(), 'Position'].values[0]
position_spent_text = position_overall_df.loc[position_overall_df['MoneySpent'] == position_overall_df['MoneySpent'].max(), 'MoneySpent'].values[0]
position_bids_text = position_overall_df.loc[position_overall_df['MoneySpent'] == position_overall_df['MoneySpent'].max(), 'WinningBids'].values[0]

##tab3
eliminated_most_text = eliminated_summary.loc[eliminated_summary['Times Eliminated'] == eliminated_summary['Times Eliminated'].max(), 'Name'].values[0]
eliminated_count_text = eliminated_summary.loc[eliminated_summary['Times Eliminated'] == eliminated_summary['Times Eliminated'].max(), 'Times Eliminated'].values[0]
eliminated_tie_text = eliminated_summary.loc[eliminated_summary['Times Eliminated'] == eliminated_summary['Times Eliminated'].max(), 'Times Eliminated'].shape[0]

released_most_text = released_summary.loc[released_summary['Times Dropped'] == released_summary['Times Dropped'].max(), 'Name'].values[0]
released_count_text = released_summary.loc[released_summary['Times Dropped'] == released_summary['Times Dropped'].max(), 'Times Dropped'].values[0]
released_tie_text = released_summary.loc[released_summary['Times Dropped'] == released_summary['Times Dropped'].max(), 'Times Dropped'].shape[0]


##################testing delete later




##################notes
#probably dont need to go with a bump chart because I can just plot ranks in plotly...moving average is the only one that makes sense
#scatterplots of acquisitions and money spent by manager/Position/week is probaby not interesting enough to plot

###from last years report:
##would love to do a area bump chart for remaining budget. it just looks better than lines
##for manager table, add total acquisitions, weeks alive, average money spent per week, avg acq count per week, and max spent in a week
##do I care about max player price by Position per week? probably not
##add avg spent by Position/manager/week to charts on the waivers tab
##for the player table, add count of times picked up on waviers, max spent, avg spent

#can I find a place for a Race Chart?
#what about a ridgeplot?
#general note - should I add a player's team to some of these tables?
#some players are slipping through the cracks on waiver summary charts
#it would be really cool to compare waiver price vs rest of season point totals for individual players - best values
#how about a scatterplot that shows points scored vs money spent cumulatively? idk
# can I do anything with possible points? Who actually should have lost each week?
# make a chart for multi-waiver players that compares their original price to subsequent prices; can also do this for number of bids"
# call out the most weekly wins, the most second places, the closest gaps between losing and surviging, the average and median gap ahead of last place, the most easy wins(at least x points ahead of last)")
# can I do something where I look at money spent on guys that were eventually dropped? Not for eliminated teams, but waiver/free agent moves where you drop a guy you spent big on.")
# Show a chart with average and median gap for winning and losing bids...is that even interesting?
# show table of bids by manager - who has bid the most, who has won the most, lost the most, who has been most active (multiple bids for same player etc); who has laid the most money out, regardless of win/loss


############################################################################################################

with tab1:
   st.header("Overall")
   st.write("We're into Week {theweek} and {teamcount} teams are still alive. The remaining overall budget has gone from 18K to {thebudget}."\
         .format(theweek=currentweek,teamcount=alive_text,thebudget=budget_left_text))
   st.plotly_chart(week_budget_chart, theme=None) ##reformat labels
   st.write("Here's how things have shaken out so far, with green being the better weeks and red the worst.")
   st.write("We thank {losers} for joining the league and letting us take their best players."\
         .format(losers=lost_teams_text))
   st.dataframe(all_matchups_wide.style.background_gradient(cmap=cm,axis=None)) ##need to figure out how to fit this all in without scrolling
   st.write("The best thing you can do in this league is score points. And when you can do so without blowing your budget, even better!",\
    "Combining the remaining budget and 3-week scoring average, {top} is atop the power rankings while {bottom} as at the bottom."\
        .format(top=pr_top,bottom=pr_bottom))
   st.dataframe(power_rankings.style.background_gradient(cmap=cm_power),hide_index=True)
   if most_points_text == most_rpoints_text:
    st.write("Below are charts showing points by week, rolling average, and cumulative. {mostpoints} has scored the most points so far and currently has the highest 3-week rolling average!"\
             .format(mostpoints=most_points_text))
   else:
    st.write("Below are charts showing points by week, rolling average, and cumulative. {mostpoints} has scored the most points so far, while {rolling} has the best 3-week rolling average."\
             .format(mostpoints=most_points_text,rolling=most_rpoints_text))
   line = st.radio("Choose Metric:", ['Points','3-Week Rolling Avg','Rolling Rank','Cumulative Points'])
   weekly_scoring_chart = px.line(all_matchups, x="Week", y=line, color="Manager",markers=True).update_layout(title="Manager "+line+" by Week")
   st.plotly_chart(weekly_scoring_chart, theme=None)
   st.write("As the season has progressed, every team has steadily gotten better. The boxplots below show how the scores have been distributed each week. The average goes up, but so does the lowest score.")
   st.plotly_chart(weekly_dist, theme=None)

with tab2:
   st.header("Waivers")
   st.write("It's great to be leading the way in remaining budget, as {budgetleader} currently is,"\
         " but that also means other teams have already bolstered their roster. It's a risky game to play."\
         .format(budgetleader=most_budget_text))
   st.plotly_chart(week_manager_budget, theme=None)
   st.write("Let's take a closer look at how waivers have gone this season. You can use the radio button to view money spent or number of winning bids.")
   bar = st.radio("Choose Metric:", ['MoneySpent','WinningBids','MaxPlayer'])
   st.write("The manager spend chart shows when and how much each manager has spent on shiny new toys.")
   week_manager_chart = px.bar(week_manager_df, x="week", y=bar, color="Manager").update_layout(title="Manager "+bar+" by Week")
   week_position_chart = px.bar(week_position_df, x="week", y=bar, color="Position").update_layout(title="Position "+bar+" by Week")
   manager_position_chart = px.bar(manager_position_df, x="Manager", y=bar, color="Position").update_layout(title="Manager "+bar+" by Position")
   position_overall_chart = px.bar(position_overall_df, x="Position", y=bar,text_auto='.2s').update_layout(title=bar+" by Position")
   st.plotly_chart(week_manager_chart, theme=None)
   st.write("The {position} position has had the most money thrown its way, with {money} spent on {bids} waiver claims."\
         .format(position=max_position_text,money=position_spent_text,bids=position_bids_text))
   st.plotly_chart(position_overall_chart, theme=None)
   st.write("The position by manager chart shows how each manager has allocated their budget. The position by week chart follows closely to big names being dropped, especially at the quarterback and tight end positions.")
   st.plotly_chart(manager_position_chart, theme=None)
   st.plotly_chart(week_position_chart, theme=None)
   
   
with tab3:
   st.header("Players")
   if eliminated_tie_text>1:
    st.write("{count} players have found themselves on the last place team {number} times. Maybe they were the problem?"\
         .format(count=eliminated_tie_text,number=eliminated_count_text))
   else:
    st.write("{player} leads the way with {number} times being dropped from the last place team. Yikes."\
         .format(player=eliminated_most_text,number=eliminated_count_text))
   st.dataframe(eliminated_summary.style, hide_index=True) ##filter for more than one elimination eventually...for now it's ok to have all
   if released_tie_text>1:
    st.write("{count} players have been dropped a total of {number} times. These players are obviously good enough to be rostered but don't stick on a roster too long."\
         .format(count=released_tie_text,number=released_count_text))
   else:
    st.write("{player} leads the way with {number} times being dropped. Will anyone else take a chance on him?"\
         .format(player=released_most_text,number=released_count_text))

   st.dataframe(released_summary.style, hide_index=True) ##filter for more than one release eventually...for now it's ok to have all
   st.write("The tree chart below shows the top acquisitions by position.") #Call out the top players for each, or maybe add a table below.
   st.plotly_chart(player_tree) 
   st.dataframe(adds_player.style, hide_index=True) ## sort options: by difference to find closest and furthest...do callouts (wides and narrowest bid gaps)
   st.dataframe(adds_player_summary.style,hide_index=True)
   st.plotly_chart(waiver_scatter)

with tab4:
   st.header("Managers")
   bar = st.radio("Choose Metric:", ['AvgWinGap','MedianWinGap','MaxWinGap','MinWinGap']) ##where does this go?
   test2_chart = px.bar(bids_winning, x="Winning Manager", y=bar,text_auto='.2s').update_layout(title=bar+" by Manager",barmode='stack', xaxis={'categoryorder':'total descending'})
   st.plotly_chart(test2_chart, theme=None)   

with tab5:
   st.header("TABLES!")
   st.image('https://44.media.tumblr.com/9a7e3822dd2771c1e7965542d7168ab1/b0ea9792e807e275-a6/s540x810_f1/e2bc4d208650b5c6a1751433189a529489e44df2.gif')
   st.write("Various tables/summaries that I'll organize later.")
   st.dataframe(rosters.style, hide_index=True)
   st.write(all_matchups)
   st.write("waivers")
   st.dataframe(week_overall_df.style, hide_index=True)
   st.dataframe(position_overall_df.style, hide_index=True)
   st.dataframe(manager_overall_df.style, hide_index=True)
   st.dataframe(week_manager_df.style, hide_index=True)
   st.dataframe(week_position_df.style, hide_index=True)
   st.dataframe(manager_position_df.style, hide_index=True)
   st.write("adds")
   st.dataframe(transactions_df.style, hide_index=True)
   st.dataframe(adds_df.style, hide_index=True)
   st.dataframe(adds_df_combined.style, hide_index=True)

