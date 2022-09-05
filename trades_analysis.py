import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
df = pd.read_csv('market_logs.log', sep = ',', header = None, names = ['one', 'two', 'three'])
df1 = pd.read_csv('exec_logs.log', sep = ',', header = None, names = ['one', 'two', 'three', 'four', 'five', 'six'])

#PRIMARY DATA SORTING OUT

#market_logs.log data sorting out
market_df = pd.DataFrame(
  { 'time' :      [ x.strip().split(' ')[4] for x in df['one'] ],
    'direction' : [ x.strip().split(' ')[7] for x in df['one'] ],
    'price' :     [ x.strip().split(' ')[1] for x in df['two'] ],
    'volume' :    [ x.strip().split(' ')[2] for x in df['three'] ],
    'vol_bid' :   [ x.strip().split('@')[0]
                      for x in [ x.strip().split(' ')[6]
                                for x in df['three'] ]],
    'pric_bid' :    [ x.strip().split('x')[0]
                     for x in [ x.strip().split('@')[1]
                      for x in [ x.strip().split(' ')[6]
                                for x in df['three'] ] ] ],
    'pric_ask' :   [ x.strip().split('x')[1]
                     for x in [ x.strip().split('@')[1]
                      for x in [ x.strip().split(' ')[6]
                                for x in df['three'] ] ] ],
    'vol_ask' :    [ x.strip().split('@')[2]
                      for x in [ x.strip().split(' ')[6]
                                for x in df['three'] ] ] }
                  )
#convert to correct data type
market_df.time = pd.to_numeric(market_df.time)
market_df.time = pd.to_datetime(market_df.time, unit = 'ns')
market_df.direction = pd.to_numeric(market_df.direction)
market_df.price = pd.to_numeric(market_df.price)
market_df.volume = pd.to_numeric(market_df.volume)
market_df.vol_bid = pd.to_numeric(market_df.vol_bid)
market_df.pric_bid = pd.to_numeric(market_df.pric_bid)
market_df.pric_ask = pd.to_numeric(market_df.pric_ask)
market_df.vol_ask = pd.to_numeric(market_df.vol_ask)
#split 'time' to date,hour,minute etc.
market_df[ 'date' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[0] ) )
market_df[ 'date' ] = pd.to_datetime(market_df[ 'date' ])
market_df[ 'hour' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[1].split(':')[0] ) ).astype(int)
market_df[ 'minute' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[1].split(':')[1] ) ).astype(int)
market_df[ 'second' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[1].split(':')[2].split('.')[0] ) ).astype(int)
market_df[ 'millisec' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split('.')[1][0:3:1] ) ).astype(int)
market_df[ 'microsec' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split('.')[1][3:6:1] ) ).astype(int)
market_df[ 'nanosec' ] = market_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split('.')[1][6:9:1] ) ).astype(int)

#exec_logs.log data sorting out
orders_df = pd.DataFrame(
  { 'time' :      [ x.strip().split(' ')[4] for x in df1['one'] ],
    'id' :        [ x.strip().split(' ')[7] for x in df1['one'] ],
    'side' :      [ x.strip().split(' ')[1] for x in df1['two'] ],
    'price' :     [ x.strip().split(' ')[1] for x in df1['three'] ],
    'volume' :    [ x.strip().split(' ')[2] for x in df1['four'] ],
    'volume_left':[ x.strip().split(' ')[4] for x in df1['five'] ],
    'delta_exec' :[ x.strip().split(' ')[1] for x in df1['six'] ]
     }
                  )
#convert to correct data type
orders_df.time = pd.to_numeric(orders_df.time)
orders_df.time = pd.to_datetime(orders_df.time, unit = 'ns')
orders_df.id = pd.to_numeric(orders_df.id)
orders_df.side = pd.to_numeric(orders_df.side)
orders_df.price = pd.to_numeric(orders_df.price)
orders_df.volume = pd.to_numeric(orders_df.volume)
orders_df.volume_left = pd.to_numeric(orders_df.volume_left)
orders_df.delta_exec = pd.to_numeric(orders_df.delta_exec)
#split 'time' to date,hour,minute etc.
orders_df[ 'date' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[0] ) )
orders_df[ 'date' ] = pd.to_datetime(orders_df[ 'date' ])
orders_df[ 'hour' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[1].split(':')[0] ) ).astype(int)
orders_df[ 'minute' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[1].split(':')[1] ) ).astype(int)
orders_df[ 'second' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split(' ')[1].split(':')[2].split('.')[0] ) ).astype(int)
orders_df[ 'millisec' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split('.')[1][0:3:1] ) ).astype(int)
orders_df[ 'microsec' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split('.')[1][3:6:1] ) ).astype(int)
orders_df[ 'nanosec' ] = orders_df[ 'time' ].astype( str ).map( lambda s: ''.join( s.split('.')[1][6:9:1] ) ).astype(int)

#TASK 2.2
#GENERAL PART
#merge market and exec tables, calculation of dmin5 column and correlation
market_without_5sec = market_df.loc [ market_df['time'] >= market_df['time'][0] + pd.Timedelta('5 seconds') ]
markets_orders = pd.merge(market_without_5sec, orders_df, how = 'right', on = ('date', 'hour','minute') )
markets_orders = markets_orders.drop_duplicates(subset = ['id'])
markets_orders = markets_orders.reset_index(drop = True)
markets_orders = markets_orders.assign( dmin5 = ( ( ( ( markets_orders.pric_bid +
                                           markets_orders.pric_bid ) /2 ) -
                                      markets_orders.price_y ) * markets_orders.side ) )

print('Correlation b/n delta_exec and dmin5\n',markets_orders.corr().unstack()['delta_exec'].iloc[23])
#answer: corr(dmin5, delta_exec) = 0.011877

#TASK 2.1

#Researching of market trades
#

#PROFIT/LOSS
#profit/loss result for period
#quantity of +/- trades
#close_trade.to_list() - open_trade.to_list()

############data sorting out
df = markets_orders
open_trade = ( df[::2]['price_y'] * df[::2]['volume_y'] * df[::2]['side'] ).reset_index(drop = True)
close_trade = ( df[1::2]['price_y'] * df[1::2]['volume_y'] * df[1::2]['side']).reset_index(drop = True)
side_trade = df[::2]['side'].reset_index(drop = True)
week_day = df[::2]['time_y'].apply(lambda x: x.weekday()).reset_index(drop = True)
hour_trade = df[::2]['time_y'].dt.hour.reset_index(drop = True)
vol_left_op = df[::2]['volume_left'].reset_index(drop = True)
vol_left_cl = df[::2]['volume_left'].reset_index(drop = True)
pos_time = df[1::2]['time_y'].reset_index(drop = True) - df[::2]['time_y'].reset_index(drop = True)
summary_data = pd.DataFrame( {'open' : open_trade, 
                            'close' : close_trade, 
                            'side' : side_trade,
                            'week_day' : week_day,
                            'hour_trade' : hour_trade,
                            'vol_left_op' : vol_left_op,
                            'vol_left_cl' : vol_left_cl,
                             'pos_time' : pos_time} )
summary_data = summary_data.assign(prof_loss = (-summary_data.close - summary_data.open) )
summary_data['pos_neg'] = pd.cut( summary_data['prof_loss'], 
        [-np.inf, -0.01, 0.01, np.inf], 
        labels = ['pos','zero','neg'] )
#open - buy price
#close - sell price
#side - long/short
#week_day - day of the week the trade was executed
#hour_trade - hour of the day the trade was executed
#vol_left_op - volume left after trade opening
#vol_left_cl- volume left after trade closing
#prof_loss - financial result of the trade
#pos_neg - profit, neutral or loss of trade
#pos_time - time which continues in the trade

# TASK 2.1
##GENERAL PART
# Data for Researching of Trades
# the simplest metrics

print( '1.Trade Result', summary_data['prof_loss'].sum( ).astype( float ).round( 2 ) )

print( '\n2.Profit Factor', (summary_data.loc[summary_data['prof_loss'] > 0,]['prof_loss'].sum( ).round( ) /
                             abs(
                                 summary_data.loc[summary_data['prof_loss'] < 0,]['prof_loss'].sum( ).round( ) )).round(
    2 )
       )

print( '\n3.Quantity of trades\n', summary_data.groupby( 'pos_neg', as_index=False ) \
       .agg( {'pos_neg': 'count'} ) \
       .rename( columns={'pos_neg': 'trades'},
                index={0: 'positive',
                       1: 'neutral',
                       2: 'negative'} ) )

print( '\n4.Long/Short Trades',
       '\n long', summary_data.loc[summary_data['side'] == 1].count( )[0],
       summary_data.loc[summary_data['side'] == 1].groupby( 'pos_neg', as_index=False ) \
       .agg( {'pos_neg': 'count'} ) \
       .rename( columns={'pos_neg': ''},
                index={0: '   positive',
                       1: '   neutral',
                       2: '   negative'} ),
       '\n short', summary_data.loc[summary_data['side'] == -1].count( )[0],
       summary_data.loc[summary_data['side'] == -1].groupby( 'pos_neg', as_index=False ) \
       .agg( {'pos_neg': 'count'} ) \
       .rename( columns={'pos_neg': ''},
                index={0: '   positive',
                       1: '   neutral',
                       2: '   negative'} )
       )

print( '\n5.Trades by day of the week\n', summary_data.groupby( ['week_day'], as_index=False ) \
       .agg( {'open': 'count'} ) \
       .rename( columns={'open': 'trades'},
                index={0: 'Monday',
                       1: 'Tuesday',
                       2: 'Wednesday',
                       3: 'Thursday',
                       4: 'Friday'} )['trades'] )

print( summary_data.groupby( ['week_day', 'pos_neg'], as_index=False ) \
       .agg( {'open': 'count'} ) \
       .rename( columns={'open': 'trades'} ) \
       .pivot( index='pos_neg', columns='week_day', values='trades' ) \
       .rename( columns={0: 'Monday',
                         1: 'Tuesday',
                         2: 'Wednesday',
                         3: 'Thursday',
                         4: 'Friday'} )
       )

print( '\n6. Trades by hour of day\n',
       summary_data.groupby( 'hour_trade', as_index=False ) \
       .agg( {'open': 'count'} ) \
       .rename( columns={'open': 'trades'} ) )

print( summary_data.groupby( ['hour_trade', 'pos_neg'], as_index=False ) \
       .agg( {'open': 'count'} ) \
       .rename( columns={'open': 'trades'} ) \
       .pivot( index='pos_neg', columns='hour_trade', values='trades' )
       )

print( '\n7. Time in the Trades\n' )
print( 'Mean Time in the position\n', summary_data['pos_time'].describe( ).iloc[[1, 3, 7]] )
print( '\nMean Time in the position of Negative Trades\n',
       summary_data.loc[summary_data['pos_neg'] == 'neg'].mean( )['pos_time'] )
print( 'Mean Time in the position of Positive Trades\n',
       summary_data.loc[summary_data['pos_neg'] == 'pos'].mean( )['pos_time'] )

result = sm.ols(formula = ' prof_loss ~ side + week_day + hour_trade ', data = summary_data).fit()
print(result.summary())