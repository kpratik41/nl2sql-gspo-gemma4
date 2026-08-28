CREATE TABLE weather (
    `station_nbr` integer, -- station number. the id of weather stations.
    `date` date, -- date.
    `tmax` integer, -- temperature max. max temperature.
    `tmin` integer, -- temperature min. min temperature. temperature range / difference = tmax - tmin.
    `tavg` integer, -- temperature average. average temperature.
    `depart` integer, -- departure from normal. Temperature departure from the normal indicates if the dekadal average temperatures were above or below the 30-year normal. â¢ if null: the temperature is 30-year normal. â¢ if the value is positive: the temperature is above the 30-year normal,. â¢ if the value is negative: the temperature is below the 30-year normal,.
    `dewpoint` integer, -- dew point. The dew point is the temperature to which air must be cooled to become saturated with water vapor, assuming constant air pressure and water content.
    `wetbulb` integer, -- wet bulb. â¢ The wet-bulb temperature (WBT) is the temperature read by a thermometer covered in water-soaked (water at ambient temperature) cloth (a wet-bulb thermometer) over which air is passed. â¢ At 100% relative humidity, the wet-bulb temperature is equal to the air temperature (dry-bulb temperature); â¢ at lower humidity the wet-bulb temperature is lower than dry-bulb temperature because of evaporative cooling.
    `heat` integer, -- calculated heating degree.
    `cool` integer, -- calculated cooling degree.
    `sunrise` text, -- calculated sunrise.
    `sunset` text, -- calculated sunset.
    `codesum` text, -- code summarization. code summarization for the weather. â¢ PY SPRAY. â¢ SQ SQUALL. â¢ DR LOW DRIFTING. â¢ SH SHOWER. â¢ FZ FREEZING. â¢ MI SHALLOW. â¢ PR PARTIAL. â¢ BC PATCHES. â¢ BL BLOWING. â¢ VC VICINITY. â¢ - LIGHT. â¢ + HEAVY. â¢ "NO SIGN" MODERATE.
    `snowfall` real, -- snowfall. snowfall (inches AND tenths).
    `preciptotal` real, -- precipitation total. inches (240hr period ending at indicated local standard time).
    `stnpressure` real, -- station pressure.
    `sealevel` real, -- sea level.
    `resultspeed` real, -- resultant speed. resultant wind speed.
    `resultdir` integer, -- resultant direction. resultant wind direction. who degree.
    `avgspeed` real, -- average speed. average wind speed. if avgspeed is larger: much more wind.
    primary key (station_nbr, date)
);

CREATE TABLE sales_in_weather (
    `date` date, -- the date of sales.
    `store_nbr` integer, -- store number.
    `item_nbr` integer, -- item number. item / product number.
    `units` integer, -- the quantity sold of an item on a given day.
    primary key (store_nbr, date, item_nbr)
);

CREATE TABLE relation (
    `store_nbr` integer, -- store number. the id of stores.
    `station_nbr` integer, -- station number. the id of weather stations.
    primary key (store_nbr),
    foreign key (store_nbr) references sales_in_weather(store_nbr),
    foreign key (station_nbr) references weather(station_nbr)
);

