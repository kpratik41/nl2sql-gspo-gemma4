CREATE TABLE Airports (
    `Code` text, -- IATA code of the air airports.
    `Description` text, -- the description of airports.
    primary key (Code)
);

CREATE TABLE Airlines (
    `FL_DATE` text, -- flight date.
    `OP_CARRIER_AIRLINE_ID` integer, -- operator carrier airline id.
    `TAIL_NUM` text, -- tail number. plane's tail number.
    `OP_CARRIER_FL_NUM` integer, -- operator carrier flight number.
    `ORIGIN_AIRPORT_ID` integer, -- origin airport id.
    `ORIGIN_AIRPORT_SEQ_ID` integer, -- origin airport sequence id.
    `ORIGIN_CITY_MARKET_ID` integer, -- origin city market id.
    `ORIGIN` text, -- airport of origin. • the origin city could be inferred by this code:. you can refer to https://www.iata.org/en/publications/directories/code-search/?airport.search=mia. to quickly check.
    `DEST_AIRPORT_ID` integer, -- destination airport id. ID of the destination airport.
    `DEST_AIRPORT_SEQ_ID` integer, -- destination airport sequence id.
    `DEST_CITY_MARKET_ID` integer, -- destination city market id.
    `DEST` text, -- destination. Destination airport. • the dest city could be inferred by this code:. you can refer to https://www.iata.org/en/publications/directories/code-search/?airport.search=mia. to quickly check.
    `CRS_DEP_TIME` integer, -- scheduled local departure time.
    `DEP_TIME` integer, -- departure time. Flight departure time. stored as the integer.
    `DEP_DELAY` integer, -- Departure delay. Departure delay indicator. in minutes. • if this value is positive: it means this flight delays; if the value is negative, it means this flight departs in advance (-4). • if this value <= 0, it means this flight departs on time.
    `DEP_DELAY_NEW` integer, -- departure delay new. not useful.
    `ARR_TIME` integer, -- arrival time. Flight arrival time.
    `ARR_DELAY` integer, -- arrival delay. arrival delay time. in minutes. • if this value is positive: it means this flight will arrives late (delay); If the value is negative, this flight arrives earlier than scheduled. (-4). • if this value <= 0, it means this flight arrives on time.
    `ARR_DELAY_NEW` integer, -- arrival delay new. not useful.
    `CANCELLED` integer, -- Flight cancellation indicator.
    `CANCELLATION_CODE` text, -- cancellation code. C--> A: more serious reasons lead to this cancellation.
    `CRS_ELAPSED_TIME` integer, -- scheduled elapsed time.
    `ACTUAL_ELAPSED_TIME` integer, -- actual elapsed time. if ACTUAL_ELAPSED_TIME < CRS_ELAPSED_TIME: this flight is faster than scheduled; if ACTUAL_ELAPSED_TIME > CRS_ELAPSED_TIME: this flight is slower than scheduled.
    `CARRIER_DELAY` integer, -- carrier delay. minutes.
    `WEATHER_DELAY` integer, -- weather delay. delay caused by the wheather problem. minutes.
    `NAS_DELAY` integer, -- National Aviavtion System delay. delay, in minutes, attributable to the National Aviation System. minutes.
    `SECURITY_DELAY` integer, -- security delay. delay attribute to security. minutes.
    `LATE_AIRCRAFT_DELAY` integer, -- late aircraft delay. delay attribute to late aircraft. minutes.
    FOREIGN KEY (ORIGIN) REFERENCES Airports(Code),
    FOREIGN KEY (DEST) REFERENCES Airports(Code),
    FOREIGN KEY (OP_CARRIER_AIRLINE_ID) REFERENCES "Air Carriers"(Code)
);

CREATE TABLE Air Carriers (
    `Code` integer, -- the code of the air carriers.
    `Description` text, -- the description of air carriers.
    primary key (Code)
);

