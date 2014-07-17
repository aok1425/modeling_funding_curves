# %matplotlib
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import getpass

pd.options.display.mpl_style = 'default'

conn = psycopg2.connect(dbname="watsi", host="localhost", user="aok1425", password=getpass.getpass('Password? '))
cur = conn.cursor()

def get_big_table(patient_id):
	query = """
		select *
		from (
		select *, bool_and(hours_after_posted > 0) over (partition by patient) as normal_hours_after_posted
		from (
		select donor_id, patient, sum(pct_of_total) over (partition by patient order by time_of_donation) as cum_pct, sum(pct_of_total) over (partition by patient) as total_pct, extract(epoch from time_of_donation)/3600 as hours_after_posted
		from (
		select donor_id, p.id as patient, p.target_amount * .01 as "funding_amt", c.donation_amount/p.target_amount :: float as "pct_of_total", c.created_at - p.created_at as "time_of_donation"
		from contributions as c join profiles as p
		on (c.contributable_id = p.id)
		where contributable_id = {}) table1) table2) table3
		where @(total_pct-1) < 0.05 and normal_hours_after_posted = true
		order by patient, cum_pct""".format(patient_id)
	
	query = """
		select donor_id, patient, cum_amt::int, hours_after_posted, num
		from (
		select *, bool_and(hours_after_posted >= 0) over (partition by patient) as normal_hours_after_posted, max(cum_amt) over (partition by patient) as max_amt, count(*) over (partition by patient) as "num"
		from (
		select donor_id, patient, sum(donation_amt) over (partition by patient order by time_of_donation) as cum_amt, total_amt, extract(epoch from (time_of_donation - min(time_of_donation) over (partition by patient)))/3600 as hours_after_posted, extract(epoch from hrs_to_completion)/3600 as hrs_to_completion
		from (
		select donor_id, p.id as patient, c.donation_amount * .01 as "donation_amt", p.target_amount * .01 as total_amt, c.created_at - p.created_at as "time_of_donation", max(c.created_at - p.created_at) over (partition by p.id) as "hrs_to_completion"
		from contributions as c join profiles as p
		on (c.contributable_id = p.id)
		where contributable_id = {}) table1) table2) table3
		where @(max_amt-total_amt)/total_amt < 0.1 and normal_hours_after_posted = true and hrs_to_completion > 37.84
		order by patient, cum_amt;""".format(patient_id)

	cur.execute(query)
	fig = pd.DataFrame(cur.fetchall(), columns = [desc[0] for desc in cur.description])
	#fig.set_index(['hours_after_posted'], inplace=True) # won't work when i have several pts

	return fig

for num in range(800,900):
	try:
		get_big_table(num).cum_amt.astype(int).plot()
	except:
		pass


from sklearn import linear_model
clf = linear_model.Ridge (alpha = .5)

fig = get_big_table(710).reset_index()
clf.fit([[i] for i in fig.cum_amt.astype(int).values], fig.index)




def convert_to_hrs(Series):
	return Series.apply(lambda x: x / np.timedelta64(1,'h'))

# 25th percentile, 54.730918270902777
np.percentile(fig.hrs_to_completion,25) / np.timedelta64(1,'h')

# 25th percentile for hrs after posted, 37.845972
b=(fig.hrs_to_completion - fig.first_donation).apply(lambda x: x / np.timedelta64(1,'h'))
np.percentile(b,25)

from sklearn.metrics import explained_variance_score



### scrap for ipython notebook
pd.DataFrame(df_scaled).describe()
df.ix[:,'exp_error':'log_error'].describe()