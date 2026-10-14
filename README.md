# tandem-data

Citi Bike station availability recorded for
[Tandem](https://github.com/alexxu2027/tandem). Citi Bike publishes live
availability but keeps no history, so a scheduled GitHub Actions workflow in the
Tandem repo records it every 5 minutes and commits the results here hourly.

**Do not edit files here by hand.** Everything is written by
[`collect-citibike.yml`](https://github.com/alexxu2027/tandem/blob/main/.github/workflows/collect-citibike.yml).

## Layout

| Path | Contents |
| --- | --- |
| `citibike/station_status/<YYYY-MM-DD>/<HHMMSS>.csv` | One file per hourly run |
| `citibike/stations/<YYYY-MM-DD>.csv` | Station names, locations and capacity seen that day |

Dates and times are UTC. A file's name is the time its run started.

## Reading the status files

Each file starts with a full snapshot of every station. After that, a station
only gets a new row when its numbers change. To find a station's availability at
a given moment, take its most recent row at or before that moment.

Snapshots are taken every 5 minutes on the clock (:00, :05, ...). GitHub
sometimes starts runs late, so expect occasional gaps between files.

Column definitions and a script that joins the files into one CSV:
[pipelines/citibike/README.md](https://github.com/alexxu2027/tandem/blob/main/pipelines/citibike/README.md).

Source: [Citi Bike GBFS feed](https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json),
subject to the [Citi Bike data license](https://ride.citibikenyc.com/data-sharing-policy).
