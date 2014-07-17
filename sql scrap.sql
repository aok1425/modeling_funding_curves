select donor_id, created_at, donation_amount, contributable_id
from contributions
where contributable_id = 105;
limit 1;

select id, target_amount, created_at
from profiles
where id = 105
limit 1;

select sum(donation_amount)
from contributions
where contributable_id = 105;


select id, target_amount, created_at
from profiles as p
where id = 105;


select c.donor_id, p.id as patient, p.target_amount * .01 as "funding_amt", c.donation_amount/p.target_amount :: float as "pct_of_total", c.created_at - p.created_at as "time_of_donation"
from contributions as c join profiles as p
on (c.contributable_id = p.id)
where contributable_id = 707;

select *
from (
select *, bool_and(hours_after_posted > 0) over (partition by patient) as normal_hours_after_posted
from (
select donor_id, patient, sum(pct_of_total) over (partition by patient order by time_of_donation) as cum_pct, sum(pct_of_total) over (partition by patient) as total_pct, extract(epoch from time_of_donation)/3600 as hours_after_posted
from (
select donor_id, p.id as patient, p.target_amount * .01 as "funding_amt", c.donation_amount/p.target_amount :: float as "pct_of_total", c.created_at - p.created_at as "time_of_donation"
from contributions as c join profiles as p
on (c.contributable_id = p.id)
where contributable_id = 707) table1) table2) table3
where @(total_pct-1) < 0.05 and normal_hours_after_posted = true
order by patient, cum_pct

/* contains total hrs, and hr of first donation */
select donor_id, patient, cum_amt, hours_after_posted, num
from (
select *, bool_and(hours_after_posted > 0) over (partition by patient) as normal_hours_after_posted, max(cum_amt) over (partition by patient) as max_amt, count(*) over (partition by patient) as "num"
from (
select donor_id, patient, sum(donation_amt) over (partition by patient order by time_of_donation) as cum_amt, total_amt, extract(epoch from time_of_donation)/3600 as hours_after_posted, extract(epoch from hrs_to_completion)/3600 as hrs_to_completion
from (
select donor_id, p.id as patient, c.donation_amount * .01 as "donation_amt", p.target_amount * .01 as total_amt, c.created_at - p.created_at as "time_of_donation", max(c.created_at - p.created_at) over (partition by p.id) as "hrs_to_completion"
from contributions as c join profiles as p
on (c.contributable_id = p.id)
where contributable_id = 710) table1) table2) table3
where @(max_amt-total_amt)/total_amt < 0.1 and normal_hours_after_posted = true and hrs_to_completion > 37.84
order by patient, cum_amt;

select *, hrs_to_completion - first_donation as "completion_after_first_donation"
from (select p.id as patient, p.target_amount * .01 as "funding_amt", max(c.created_at - p.created_at) as "hrs_to_completion", min(c.created_at - p.created_at) as "first_donation", count(c.created_at) as "num"
from contributions as c join profiles as p
on (c.contributable_id = p.id)
group by patient) table1			
limit 10;

/* contains completion_after_first_donation */
select donor_id, patient, cum_amt::int, hours_after_posted, num
from (
select *, bool_and(hours_after_posted >= 0) over (partition by patient) as normal_hours_after_posted, max(cum_amt) over (partition by patient) as max_amt, count(*) over (partition by patient) as "num"
from (
select donor_id, patient, sum(donation_amt) over (partition by patient order by time_of_donation) as cum_amt, total_amt, extract(epoch from (time_of_donation - min(time_of_donation) over (partition by patient)))/3600 as hours_after_posted, extract(epoch from hrs_to_completion)/3600 as hrs_to_completion
from (
select donor_id, p.id as patient, c.donation_amount * .01 as "donation_amt", p.target_amount * .01 as total_amt, c.created_at - p.created_at as "time_of_donation", max(c.created_at - p.created_at) over (partition by p.id) as "hrs_to_completion"
from contributions as c join profiles as p
on (c.contributable_id = p.id)
where contributable_id = 710) table1) table2) table3
where @(max_amt-total_amt)/total_amt < 0.1 and normal_hours_after_posted = true and hrs_to_completion > 37.84
order by patient, cum_amt;

/* num of donations for each pt */
select contributable_id, count(*)
from contributions
group by contributable_id
limit 10;

select donor_id, patient, cum_amt, hours_after_posted, num
from (
	select *, bool_or(hours_after_posted < 0) as normal_hours_after_posted, max(cum_amt) over (partition by patient) as max_amt, count(*) over (partition by patient) as "num"
	from (
		select donor_id, patient, sum(donation_amt) as cum_amt, total_amt, extract(epoch from time_of_donation)/3600 as hours_after_posted, extract(epoch from hrs_to_completion)/3600 as hrs_to_completion
		from (
			select donor_id, p.id as patient, c.donation_amount * .01 as "donation_amt", p.target_amount * .01 as total_amt, c.created_at - p.created_at as "time_of_donation", max(c.created_at - p.created_at) over (partition by p.id) as "hrs_to_completion"
			from contributions as c join profiles as p
			on (c.contributable_id = p.id)
			where contributable_id = 710) table1) table2
	group by patient) table3
where @(max_amt-total_amt)/total_amt < 0.1 and normal_hours_after_posted = true and hrs_to_completion > 37.84
order by patient, cum_amt;

sum(donation_amt)
group by patient

total_amt - above

/* pulling profiles that meet filter criteria, in order to get error stats on each of the profiles */
select patient, count
from(
select p.id as patient, count(*) as count, @((sum(c.donation_amount)-p.target_amount::float)/p.target_amount) as diff_pct, p.target_amount * .01 as total_amt, extract(epoch from min(c.created_at - p.created_at))/3600 as "min_time_of_donation", extract(epoch from max(c.created_at - p.created_at))/3600 as "hrs_to_completion"
from contributions as c join profiles as p
on (c.contributable_id = p.id)
group by p.id
order by diff_pct desc) table1
where diff_pct < 0.1 and min_time_of_donation > 0 and hrs_to_completion > 37.84;


/* inspecting profiles w weird diff_pcts */
select p.id, p.target_amount, sum(c.donation_amount) over (partition by p.id order by c.donation_amount)
from contributions as c join profiles as p
on (c.contributable_id = p.id)
where p.id = 137;

/* at time of first donation for a pt, how many other pts cld be funded? */
pt_id
first_donation
posted_date
last_donation = funded_date


select table2.contributable_id as t2id, table4.contributable_id as t4id, table4.posted_date, table2.first_donation, table4.funded_date

from
(select *
from (select c.contributable_id, max(p.created_at) as posted_date, min(c.created_at) as first_donation, max(p.funded_at) as funded_date
from contributions as c join profiles as p
on c.contributable_id = p.id
group by c.contributable_id) table1
where posted_date > first_donation /* there are 42 of these */ and funded_date is not null /* 33 of these */
limit 10) table2
,
(select *
from (select c.contributable_id, max(p.created_at) as posted_date, min(c.created_at) as first_donation, max(p.funded_at) as funded_date
from contributions as c join profiles as p
on c.contributable_id = p.id
group by c.contributable_id) table3
where posted_date > first_donation /* there are 42 of these */ and funded_date is not null /* 33 of these */
limit 10) table4

where table2.first_donation < table4.funded_date and table2.first_donation > table4.posted_date;

/* table2 has the pt in question. table4 shows the other pts that were fundable at the time of the first_donation to the pt in table2' */

select *
from (select c.contributable_id, max(p.created_at) as posted_date, min(c.created_at) as first_donation, max(p.funded_at) as funded_date
from contributions as c join profiles as p
on c.contributable_id = p.id
group by c.contributable_id) table1
where posted_date > first_donation /* there are 42 of these */ and funded_date is not null /* 33 of these */
limit 10;