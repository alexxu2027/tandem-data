# Group model v1: docks at arrival (linear regression)

**Question:** "I'm riding to this station and will get there in 30 minutes. Will there be a free dock?"
Today the app shows only how many docks are free *right now*.

**Run it** (from the repo root; installs what it needs, takes about 2 minutes):

    uv run model/dock_regression.py

It writes `model/output/results.csv` and `model/output/baseline_vs_v1.png`.

| Item | Value |
| --- | --- |
| Data | Citi Bike GBFS feed, recorded every 5 min by the collector in this repo (real data, not synthetic) |
| Rows | One per station per 5-minute moment, weekdays 4-7 PM New York time |
| Split | Random 70% train / 30% test (seed 42) |
| Outcome | `docks_in_30`: free docks 30 minutes later |
| Features | `docks_now`, `docks_change_15` (change over last 15 min), `capacity`, `minutes_after_4pm` |
| Baseline | Predict the training average for every station |
| Decision rule | Predicted docks < 3 → suggest another station; otherwise "head there" |

**Value assumptions (made up, labelled as such on the slides):** a rider who arrives at a full
station loses 8 minutes; a rider sent elsewhere when there was room loses 3 minutes; rider time
is worth $0.50/minute; 10,000 evening-rush returns per weekday.

**Caveat:** rows from the same station a few minutes apart look alike, so a random split is
kinder than predicting future days. The main Tandem repo also scores models on later days.
