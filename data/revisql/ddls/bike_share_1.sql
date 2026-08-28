CREATE TABLE trip (
    `id` integer, -- No description.
    `duration` integer, -- The duration of the trip in seconds. duration = end_date - start_date.
    `start_date` text, -- start date.
    `start_station_name` text, -- start station name. The name of the start station. It represents the station name the bike borrowed from.
    `start_station_id` integer, -- start station id. The ID of the start station.
    `end_date` text, -- end date.
    `end_station_name` text, -- end station name. The name of the end station. It represents the station name the bike returned to.
    `end_station_id` integer, -- end station id. The ID of the end station.
    `bike_id` integer, -- bike id. The ID of the bike.
    `subscription_type` text, -- subscription type. Allowed input: Subscriber, Customer.
    `zip_code` integer, -- zip code.
    primary key (id)
);

CREATE TABLE weather (
    `date` text, -- No description.
    `max_temperature_f` integer, -- max temperature in Fahrenheit degree. It represents the hottest temperature.
    `mean_temperature_f` integer, -- mean temperature in Fahrenheit degree.
    `min_temperature_f` integer, -- min temperature in Fahrenheit degree. It represents the coldest temperature.
    `max_dew_point_f` integer, -- max dew point in Fahrenheit degree.
    `mean_dew_point_f` integer, -- mean dew point in Fahrenheit degree.
    `min_dew_point_f` integer, -- min dew point in Fahrenheit degree.
    `max_humidity` integer, -- max humidity.
    `mean_humidity` integer, -- mean humidity.
    `min_humidity` integer, -- min humidity.
    `max_sea_level_pressure_inches` real, -- max sea level pressure in inches.
    `mean_sea_level_pressure_inches` real, -- mean sea level pressure in inches.
    `min_sea_level_pressure_inches` real, -- min sea level pressure in inches.
    `max_visibility_miles` integer, -- max visibility in miles.
    `mean_visibility_miles` integer, -- mean visibility in miles.
    `min_visibility_miles` integer, -- min visibility in miles.
    `max_wind_Speed_mph` integer, -- max wind Speed in mph.
    `mean_wind_speed_mph` integer, -- mean wind Speed in mph.
    `max_gust_speed_mph` integer, -- max gust Speed in mph.
    `precipitation_inches` text, -- precipitation in inches.
    `cloud_cover` integer, -- cloud cover.
    `events` text, -- Allowed input: [null], Rain, other.
    `wind_dir_degrees` integer, -- wind direction degrees.
    `zip_code` text, -- zip code.

);

CREATE TABLE station (
    `id` integer, -- unique ID for each station.
    `name` text, -- Station name.
    `lat` real, -- latitude. Can represent the location of the station when combined with longitude.
    `long` real, -- longitude. Can represent the location of the station when combined with. latitude.
    `dock_count` integer, -- dock count. number of bikes the station can hold.
    `city` text, -- No description.
    `installation_date` text, -- installation date.
    primary key (id)
);

CREATE TABLE status (
    `station_id` integer, -- station id.
    `bikes_available` integer, -- number of available bikes. 0 means no bike can be borrowed.
    `docks_available` integer, -- number of available docks. 0 means no bike can be returned to this station.
    `time` text, -- No description.

);

