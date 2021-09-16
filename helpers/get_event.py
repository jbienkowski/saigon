from obspy import read

# EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName
# knmi2019jxxm|2019-05-22T03:49:00.5|53.328000|6.652000|3.0|||KNMI|knmi2019jxxm|MLn|3.361468904||Westerwijtwerd

st1 = read(
    "http://rdsa.knmi.nl/fdsnws/dataselect/1/query?starttime=2019-05-22T00%3A00%3A00&endtime=2019-05-22T23%3A59%3A59&network=NL&station=BSTD&channel=HGZ"
)
st1.plot(outfile="out/NL-BSTD-HGZ-knmi2019jxxm-daily.pdf", type="dayplot")

st2 = read(
    "http://rdsa.knmi.nl/fdsnws/dataselect/1/query?starttime=2019-05-22T03%3A49%3A00&endtime=2019-05-22T03%3A49%3A30&network=NL&station=BSTD&channel=HGZ"
)
st2.plot(outfile="out/NL-BSTD-HGZ-knmi2019jxxm-zoomed.pdf")
