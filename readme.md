
Process:
Matches trafficdata to georgaphically closest weatherstation. Uses this weatherdata, the traficdata, deflated oilprices and the holiday calender to do OLS-regressions on daytype.


Script structure:
1. API-downloads relevant raindata and Holiday Calender
2. Merges and cleanes data
3. OLS-regressions

Datasets:
Historical traffic counts from 2005-2014: "https://www.opendata.dk/city-of-copenhagen/faste-trafiktaellinger"
OK historic LeadFree95(E10)&Diesel historic pricedevelopment: "https://www.ok.dk/privat/produkter/fyringsolie/prisudvikling"
Historic weatherdata openAPI: "https://www.dmi.dk/friedata/dokumentation/apis"
Holidaycalender API: "https://date.nager.at/Api?utm_source=chatgpt.com"
CPI Statistics Denmark PRIS01: "https://www.statbank.dk/20072"