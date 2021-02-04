# flake8: noqa

# In[]:
# Controls for webapp
COUNTIES = {
    "001": "Albany",
    "003": "Allegany",
    "005": "Bronx",
    "007": "Broome",
    "009": "Cattaraugus",
    "011": "Cayuga",
    "013": "Chautauqua",
    "015": "Chemung",
    "017": "Chenango",
    "019": "Clinton",
    "021": "Columbia",
    "023": "Cortland",
    "025": "Delaware",
    "027": "Dutchess",
    "029": "Erie",
    "031": "Essex",
    "033": "Franklin",
    "035": "Fulton",
    "037": "Genesee",
    "039": "Greene",
    "041": "Hamilton",
    "043": "Herkimer",
    "045": "Jefferson",
    "047": "Kings",
    "049": "Lewis",
    "051": "Livingston",
    "053": "Madison",
    "055": "Monroe",
    "057": "Montgomery",
    "059": "Nassau",
    "061": "New York",
    "063": "Niagara",
    "065": "Oneida",
    "067": "Onondaga",
    "069": "Ontario",
    "071": "Orange",
    "073": "Orleans",
    "075": "Oswego",
    "077": "Otsego",
    "079": "Putnam",
    "081": "Queens",
    "083": "Rensselaer",
    "085": "Richmond",
    "087": "Rockland",
    "089": "St. Lawrence",
    "091": "Saratoga",
    "093": "Schenectady",
    "095": "Schoharie",
    "097": "Schuyler",
    "099": "Seneca",
    "101": "Steuben",
    "103": "Suffolk",
    "105": "Sullivan",
    "107": "Tioga",
    "109": "Tompkins",
    "111": "Ulster",
    "113": "Warren",
    "115": "Washington",
    "117": "Wayne",
    "119": "Westchester",
    "121": "Wyoming",
    "123": "Yates",
}

WELL_STATUSES = dict(
    Monday="Monday",
    Tuesday="Tuesday",
    Wednesday="Wednesday",
    Thursday="Thursday",
    Friday="Friday",
    Saturday="Saturday",
    Sunday="Sunday",
)

WELL_TYPES = dict(
    F12="F12",
     _96="96",
    F31="F31",
    _367="367",
    F47="F47",
    C14="C14",
    _22="22",
    _15="15",
    F20="F20",
    F10="F10",
    F09="F09",
    _10="10",

)

WELL_COLORS = dict(
    GD="#FFEDA0",
    GE="#FA9FB5",
    GW="#A1D99B",
    IG="#67BD65",
    OD="#BFD3E6",
    OE="#B3DE69",
    OW="#FDBF6F",
    ST="#FC9272",
    BR="#D0D1E6",
    MB="#ABD9E9",
    IW="#3690C0",
    LP="#F87A72",
    MS="#CA6BCC",
    Confidential="#DD3497",
    DH="#4EB3D3",
    DS="#FFFF33",
    DW="#FB9A99",
    MM="#A6D853",
    NL="#D4B9DA",
    OB="#AEB0B8",
    SG="#CCCCCC",
    TH="#EAE5D9",
    UN="#C29A84",
)

# DIRECTIONS=dict(
#     Direction_1 ="Direction 1",
#     Direction_2 ="Direction 2"
# )

# PREDICTED_DESTINATION = {
#     "International City, Entrance EA"	:"International City, Entrance EA",
# "Manama Road 4 2"	:"Manama Road 4 2",
#     "Al Warsan 1, Eppco 1"	:"Al Warsan 1, Eppco 1",
# "International City, COSCO Logistics 1"	:"International City, COSCO Logistics 1",
# "International City, COSCO Logistics 2"	:"International City, COSCO Logistics 2",
# "International City, Dragon Mart 2"	:"International City, Dragon Mart 2",
# "International City, ibis Styles Hotel"	 : "International City, ibis Styles Hotel",
# "International City, Entrance EA"	:"International City, Entrance EA",
# "Manama Road 4 1"	:"Manama Road 4 1",
# "Downtown Mirdiff Gate 2 2"	:"Downtown Mirdiff Gate 2 2",
# "Mirdiff Area, Street 20C"	:"Mirdiff Area, Street 20C",
#
# }

TIME_ZONES ={
    "4AM_7AM":"4AM_7AM",
    "7AM_10AM":"7AM_10AM",
    "10AM_1PM":"10AM_1PM",
    "1PM_4PM":"1PM_4PM",
    "4PM_7PM":"4PM_7PM",
    "7PM_10PM":"7PM_10PM",
    "10PM_1AM":"10PM_1AM"
}

import pandas as pd
import pathlib

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

PREDICTED_DESTINATION ={}
df_predictions = pd.read_csv(DATA_PATH.joinpath("route_367.csv"))
df_predictions = df_predictions[df_predictions['Cluster']=="Underused"]["Stop_name"]
for row in df_predictions:
    PREDICTED_DESTINATION[str(row)] = row