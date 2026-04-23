 # [Copenhagen Trafficelasticity by Oilprice 2011-2013](https://kgspindler.live/traffic_gasoline_analysis.html)

## Results:
This analysis examines whether oil-fuel price fluctuations are associated with lower road traffic in Copenhagen, by using traffic counting station data from 2011 to 2013. In a model with road fixed effects, weekday controls, local rainfall, and a smooth time trend, the estimated elasticity of traffic with respect to the real gasoline price is about −0.559. This implies that a 1% increase in the real gasoline price decreases the traffic amount by about 0.56%. Alternative specifications with lagged and further smoothed fuel prices suggest that the negative relationship remains present in several robustness checks, although its size depends on how fuel prices are timed and smoothed. The model is therefore somewhat specification sensitive, but the relatively large estimates may in part reflect Copenhagen's strong substitution opportunities, including cycling and public transport.

## Method

The historical traffic counts from Copenhagen’s fixed counting points in 2011 to 2013 are matched to historical weather data from the nearest DMI weather station. Public holidays and holiday-adjacent days are removed to avoid atypical traffic patterns. Fuel prices are merged by date, and real gasoline prices are constructed using Statistics Denmark’s CPI (`PRIS1`).

The main model is:

$$
\log(\text{Traffic}_{it})
= \alpha_i
+
\delta_{d(t)}
+
\beta \log(RGP_t)
+
\gamma \,\text{Rain}_{it}
+
\theta_1 t
+
\theta_2 t^2
+
\varepsilon_{it}
$$

where:

- $\text{Traffic}_{it}$ is the daily traffic count at road counting station $i$ on day $t$
- $\alpha_i$ are road fixed effects, capturing time-invariant differences across road counting stations
- $\delta_{d(t)}$ are weekday fixed effects, capturing systematic differences between Mondays, Tuesdays, ..., Sundays
- $RGP_t$ is the real gasoline price on day $t$
- $\text{Rain}_{it}$ is daily rainfall at the weather station matched to road station $i$ on day $t$
- $t$ is a linear time trend
- $t^2$ is a quadratic time trend, allowing for non-linear development over time
- $\varepsilon_{it}$ is the error term

In this log-log specification, the coefficient $\beta$ is interpreted as the elasticity of traffic with respect to the real gasoline price.

Additional robustness checks use lagged and moving-average fuel prices, alternative time controls, and a city-day specification. The regression analysis is restricted to 2011–2013, because changes in traffic measurement rules and coverage from 2014 onward lowered recorded traffic counts and reduced comparability with earlier years.


## Datasets:
Historical traffic counts from 2005-2014: "https://www.opendata.dk/city-of-copenhagen/faste-trafiktaellinger"

OK historic LeadFree95(E10)&Diesel historic pricedevelopment: "https://www.ok.dk/privat/produkter/fyringsolie/prisudvikling"

Historic weatherdata openAPI: "https://www.dmi.dk/friedata/dokumentation/apis"

Holidaycalender API: "https://date.nager.at/Api?utm_source=chatgpt.com"

CPI Statistics Denmark PRIS01: "https://www.statbank.dk/20072"
