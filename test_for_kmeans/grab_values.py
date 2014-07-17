# either i can combine everything in pandas, or in SQL. i shld try both!

import pandas as pd, numpy as np, getpass, psycopg2
#import pg
from math import log
#conn = pg.connect(dbname="postgres", host="localhost", user="postgres")
#conn = pg.connect(dbname="watsi", host="localhost", user="postgres", passwd=raw_input('Password? '))
conn = psycopg2.connect(dbname="watsi", host="localhost", user="aok1425", password=getpass.getpass('Password? '))
cur = conn.cursor()


### Feature 1
cur.execute("select donor_id, date_part('day',current_date-min(created_at)), sum(donation_amount)*.01 as s from contributions group by donor_id")
table1=pd.DataFrame(cur.fetchall())
table1=table1.set_index(table1[0])
table1[3]=table1[2].astype('float64')/table1[1]
table1=table1[3].sort_index() #series

### Feature 2
# doesn't take into account how long from today since their last donation. to do that, can use feature 3 and take total number of days since start/number of donations

table2=pd.read_pickle('./test_for_kmeans/table 2 from kmeans.pkl')

"""
# subtracting one date from the subsequent one for each donor_id
cur.execute("select donor_id, date_trunc('day',created_at) as date from contributions order by donor_id, date")
table2a=pd.DataFrame(cur.fetchall(),columns=['ID','date donated'])
table2a['date donated']=table2a['date donated'].astype('datetime64[ns]')
hier=pd.Series(table2a['date donated'].values,index=[table2a['ID'].values,table2a.index.values])

# if I cld find the diff btwn one value and the next for each donor_id using just pandas, that wld be awesome
thedict={}
for id in hier.index.levels[0]:
	print 'Running through patient',id,'out of',len(hier.index.levels[0])
	list=hier[id].tolist()
	state1=0
	for item in range(1,len(hier[id])):
		diff1=list[item]-list[item-1]
		diff=diff1.days
		state1+=diff
		state=state1/float(len(hier[id])-1)
	thedict[id]=state
table2=pd.Series(thedict)
"""

### Feature 3: # times they donated/day since their first donation
# takes into account how long since last donation
query3="select donor_id,thecount/days as freq from(select donor_id, count(*) as thecount, date_part('day',current_date-min(created_at)) as days from contributions group by donor_id) t1 order by freq desc"
cur.execute(query3)
table3=pd.DataFrame(cur.fetchall())
table3=table3.set_index(table3[0])
table3=table3.drop([0],axis=1)
table3=table3.sort_index()[1]

### Feature 4: Tip as percentage of donation
query4="select donor_id, tip/donation as tips_as_pcg_of_donation from(select donor_id, sum(tip_amount)*.01 as tip, sum(donation_amount)*.01 as donation from contributions group by donor_id) t1"
cur.execute(query4)
table4=pd.DataFrame(cur.fetchall())
table4=table4.set_index(table4[0])
table4=table4.drop([0],axis=1)
table4=table4.sort_index()[1]
table4=table4.astype('float64')

### Feature 5: avg % of each pt they funded, giving them major points if they fully funded a pt or funded mowt of that ptd
# put it on a log curve!

def thenorm(table):
	return (table-table.mean())/table.std()

def scale(t):
	"""Does feature scaling to [-1,1] for a table or a vertical array in a table.."""
	top=2*t-t.max()-t.min()
	bottom=t.max()-t.min()
	return top/bottom

def ci(t):
	"""t is a column from the bigtable"""
	t.name='value'
	border2=int(t.size*.9) # always 'rounds down'
	border1=int(t.size*.1)
	t=t.order()
	t=t.reset_index() # so i can take the 5th and 95th percentile values
	cut1=t[t.index>border1]
	cut2=cut1[cut1.index<border2]
	cut3=cut2.set_index(cut2.ix[:,0])
	cut4=cut3['value']
	return cut4

def thelog(t):
	t=t+1e-6
	t=log(t)
	return t
	
prenorm=pd.concat([table1,table2,table3,table4],axis=1)
#prenorm=pd.concat([ci(table1),ci(table2),ci(table3),ci(table4)],axis=1)
# this cuts off the outliers before normalization and feature scaling
prenorm+=1e-6
prenorm=prenorm.applymap(lambda x:log(x))
norm=thenorm(prenorm)
scaled=scale(norm)
values=scaled.values