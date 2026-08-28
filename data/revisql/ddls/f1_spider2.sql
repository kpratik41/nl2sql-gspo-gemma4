CREATE TABLE "circuits" (
  "circuit_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "circuit_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['albert_park', 'sepang', 'bahrain']
  "name" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Albert Park Grand Prix Circuit', 'Sepang International Circuit', 'Bahrain International Circuit']
  "location" VARCHAR(255) DEFAULT NULL, -- example: ['Melbourne', 'Kuala Lumpur', 'Sakhir']
  "country" VARCHAR(255) DEFAULT NULL, -- example: ['Australia', 'Malaysia', 'Bahrain']
  "lat" FLOAT DEFAULT NULL, -- example: [-37.8497, 2.76083, 26.0325]
  "lng" FLOAT DEFAULT NULL, -- example: [144.968, 101.738, 50.5106]
  "alt" INT(11) DEFAULT NULL, -- example: [10, 18, 7]
  "url" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['http://en.wikipedia.org/wiki/Melbourne_Grand_Prix_Circuit', 'http://en.wikipedia.org/wiki/Sepang_International_Circuit', 'http://en.wikipedia.org/wiki/Bahrain_International_Circuit']
  PRIMARY KEY ("circuit_id")
)

CREATE TABLE "constructor_results" (
  "constructor_results_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "race_id" INT(11) NOT NULL DEFAULT '0', -- example: [18, 19, 20]
  "constructor_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "points" FLOAT DEFAULT NULL, -- example: [14.0, 8.0, 9.0]
  "status" VARCHAR(255) DEFAULT NULL, -- example: ['D']
  PRIMARY KEY ("constructor_results_id")
)

CREATE TABLE "constructor_standings" (
  "constructor_standings_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "race_id" INT(11) NOT NULL DEFAULT '0', -- example: [18, 19, 20]
  "constructor_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "points" FLOAT NOT NULL DEFAULT '0', -- example: [14.0, 8.0, 9.0]
  "position" INT(11) DEFAULT NULL, -- example: [1, 3, 2]
  "position_text" VARCHAR(255) DEFAULT NULL, -- example: ['1', '3', '2']
  "wins" INT(11) NOT NULL DEFAULT '0', -- example: [1, 0, 2]
  PRIMARY KEY ("constructor_standings_id")
)

CREATE TABLE "constructors" (
  "constructor_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "constructor_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['mclaren', 'bmw_sauber', 'williams']
  "name" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['McLaren', 'BMW Sauber', 'Williams']
  "nationality" VARCHAR(255) DEFAULT NULL, -- example: ['British', 'German', 'French']
  "url" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['http://en.wikipedia.org/wiki/McLaren', 'http://en.wikipedia.org/wiki/BMW_Sauber', 'http://en.wikipedia.org/wiki/Williams_Grand_Prix_Engineering']
  PRIMARY KEY ("constructor_id")
)

CREATE TABLE "driver_standings" (
  "driver_standings_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "race_id" INT(11) NOT NULL DEFAULT '0', -- example: [18, 19, 20]
  "driver_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "points" FLOAT NOT NULL DEFAULT '0', -- example: [10.0, 8.0, 6.0]
  "position" INT(11) DEFAULT NULL, -- example: [1, 2, 3]
  "position_text" VARCHAR(255) DEFAULT NULL, -- example: ['1', '2', '3']
  "wins" INT(11) NOT NULL DEFAULT '0', -- example: [1, 0, 2]
  PRIMARY KEY ("driver_standings_id")
)

CREATE TABLE "drivers" (
  "driver_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "driver_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['hamilton', 'heidfeld', 'rosberg']
  "number" INT(11) DEFAULT NULL, -- example: [44, 6, 14]
  "code" VARCHAR(3) DEFAULT NULL, -- example: ['HAM', 'HEI', 'ROS']
  "forename" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Lewis', 'Nick', 'Nico']
  "surname" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Hamilton', 'Heidfeld', 'Rosberg']
  "full_name" VARCHAR(255) AS (forename || ' ' || surname) VIRTUAL, -- example: ['Lewis Hamilton', 'Nick Heidfeld', 'Nico Rosberg']
  "dob" DATE DEFAULT NULL, -- example: ['1985-01-07', '1977-05-10', '1985-06-27']
  "nationality" VARCHAR(255) DEFAULT NULL, -- example: ['British', 'German', 'Spanish']
  "url" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['http://en.wikipedia.org/wiki/Lewis_Hamilton', 'http://en.wikipedia.org/wiki/Nick_Heidfeld', 'http://en.wikipedia.org/wiki/Nico_Rosberg']
  PRIMARY KEY ("driver_id")
)

CREATE TABLE "lap_times" (
  "race_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "driver_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "lap" INT(11) NOT NULL, -- example: [1, 2, 3]
  "position" INT(11) DEFAULT NULL, -- example: [1, 3, 4]
  "time" VARCHAR(255) DEFAULT NULL, -- example: ['1:38.109', '1:33.006', '1:32.713']
  "milliseconds" INT(11) DEFAULT NULL, -- example: [98109, 93006, 92713]
  "seconds" FLOAT AS (CAST(milliseconds AS FLOAT) / 1000) VIRTUAL, -- example: [98.109, 93.006, 92.713]
  PRIMARY KEY ("race_id", "driver_id", "lap")
)

CREATE TABLE "pit_stops" (
  "race_id" INT(11) NOT NULL, -- example: [841, 842, 843]
  "driver_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "stop" INT(11) NOT NULL, -- example: [1, 2, 3]
  "lap" INT(11) NOT NULL, -- example: [1, 11, 12]
  "time" TIME NOT NULL, -- example: ['17:05:23', '17:05:52', '17:20:48']
  "duration" VARCHAR(255) DEFAULT NULL, -- example: ['26.898', '25.021', '23.426']
  "milliseconds" INT(11) DEFAULT NULL, -- example: [26898, 25021, 23426]
  "seconds" FLOAT AS (CAST(milliseconds AS FLOAT) / 1000) VIRTUAL, -- example: [26.898, 25.021, 23.426]
  PRIMARY KEY ("race_id", "driver_id", "stop")
)

CREATE TABLE "qualifying" (
  "qualify_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "race_id" INT(11) NOT NULL DEFAULT '0', -- example: [18, 19, 20]
  "driver_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 9, 5]
  "constructor_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 6]
  "number" INT(11) NOT NULL DEFAULT '0', -- example: [22, 4, 23]
  "position" INT(11) DEFAULT NULL, -- example: [1, 2, 3]
  "q1" VARCHAR(255) DEFAULT NULL, -- example: ['1:26.572', '1:26.103', '1:25.664']
  "q2" VARCHAR(255) DEFAULT NULL, -- example: ['1:25.187', '1:25.315', '1:25.452']
  "q3" VARCHAR(255) DEFAULT NULL, -- example: ['1:26.714', '1:26.869', '1:27.079']
  PRIMARY KEY ("qualify_id")
)

CREATE TABLE "races" (
  "race_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "year" INT(11) NOT NULL DEFAULT '0', -- example: [2009, 2008, 2007]
  "round" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "circuit_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 17]
  "name" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Australian Grand Prix', 'Malaysian Grand Prix', 'Chinese Grand Prix']
  "date" DATE NOT NULL, -- example: ['2009-03-29', '2009-04-05', '2009-04-19']
  "time" TIME DEFAULT NULL, -- example: ['06:00:00', '09:00:00', '07:00:00']
  "url" VARCHAR(255) DEFAULT NULL, -- example: ['http://en.wikipedia.org/wiki/2009_Australian_Grand_Prix', 'http://en.wikipedia.org/wiki/2009_Malaysian_Grand_Prix', 'http://en.wikipedia.org/wiki/2009_Chinese_Grand_Prix']
  "fp1_date" VARCHAR(255) DEFAULT NULL, -- example: ['2021-04-16', '2022-03-18', '2021-03-26']
  "fp1_time" VARCHAR(255) DEFAULT NULL, -- example: ['12:00:00', '14:00:00', '03:00:00']
  "fp2_date" VARCHAR(255) DEFAULT NULL, -- example: ['2021-04-16', '2022-03-18', '2021-03-26']
  "fp2_time" VARCHAR(255) DEFAULT NULL, -- example: ['15:00:00', '17:00:00', '06:00:00']
  "fp3_date" VARCHAR(255) DEFAULT NULL, -- example: ['2021-04-17', '2022-03-19', '2021-03-27']
  "fp3_time" VARCHAR(255) DEFAULT NULL, -- example: ['12:00:00', '14:00:00', '03:00:00']
  "quali_date" VARCHAR(255) DEFAULT NULL, -- example: ['2021-04-17', '2022-03-19', '2021-03-27']
  "quali_time" VARCHAR(255) DEFAULT NULL, -- example: ['15:00:00', '17:00:00', '06:00:00']
  "sprint_date" VARCHAR(255) DEFAULT NULL, -- example: ['2021-07-17', '2021-09-11', '2021-11-13']
  "sprint_time" VARCHAR(255) DEFAULT NULL, -- example: ['14:30:00', '19:30:00', '13:30:00']
  PRIMARY KEY ("race_id")
)

CREATE TABLE "results" (
  "result_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "race_id" INT(11) NOT NULL DEFAULT '0', -- example: [18, 19, 20]
  "driver_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "constructor_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "number" INT(11) DEFAULT NULL, -- example: [22, 3, 7]
  "grid" INT(11) NOT NULL DEFAULT '0', -- example: [1, 5, 7]
  "position" INT(11) DEFAULT NULL, -- example: [1, 2, 3]
  "position_text" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['1', '2', '3']
  "position_order" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "points" FLOAT NOT NULL DEFAULT '0', -- example: [10.0, 8.0, 6.0]
  "laps" INT(11) NOT NULL DEFAULT '0', -- example: [58, 57, 55]
  "time" VARCHAR(255) DEFAULT NULL, -- example: ['1:34:50.616', '+5.478', '+8.163']
  "milliseconds" INT(11) DEFAULT NULL, -- example: [5690616, 5696094, 5698779]
  "fastest_lap" INT(11) DEFAULT NULL, -- example: [39, 41, 58]
  "rank" INT(11) DEFAULT '0', -- example: [2, 3, 5]
  "fastest_lap_time" VARCHAR(255) DEFAULT NULL, -- example: ['1:27.452', '1:27.739', '1:28.090']
  "fastest_lap_speed" VARCHAR(255) DEFAULT NULL, -- example: ['218.300', '217.586', '216.719']
  "status_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 11, 5]
  PRIMARY KEY ("result_id")
)

CREATE TABLE "seasons" (
  "year" INT(11) NOT NULL DEFAULT '0', -- example: [1950, 1951, 1952]
  "url" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['http://en.wikipedia.org/wiki/2009_Formula_One_season', 'http://en.wikipedia.org/wiki/2008_Formula_One_season', 'http://en.wikipedia.org/wiki/2007_Formula_One_season']
  PRIMARY KEY ("year")
)

CREATE TABLE "status" (
  "status_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "status" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Finished', 'Disqualified', 'Accident']
  PRIMARY KEY ("status_id")
)

CREATE TABLE "sprint_results" (
  "result_id" INT(11) NOT NULL, -- example: [1, 2, 3]
  "race_id" INT(11) NOT NULL DEFAULT '0', -- example: [1061, 1065, 1071]
  "driver_id" INT(11) NOT NULL DEFAULT '0', -- example: [830, 1, 822]
  "constructor_id" INT(11) NOT NULL DEFAULT '0', -- example: [9, 131, 6]
  "number" INT(11) DEFAULT NULL, -- example: [33, 44, 77]
  "grid" INT(11) NOT NULL DEFAULT '0', -- example: [2, 1, 3]
  "position" INT(11) DEFAULT NULL, -- example: [1, 2, 3]
  "position_text" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['1', '2', '3']
  "position_order" INT(11) NOT NULL DEFAULT '0', -- example: [1, 2, 3]
  "points" FLOAT NOT NULL DEFAULT '0', -- example: [3.0, 2.0, 1.0]
  "laps" INT(11) NOT NULL DEFAULT '0', -- example: [17, 16, 18]
  "time" VARCHAR(255) DEFAULT NULL, -- example: ['25:38.426', '+1.430', '+7.502']
  "milliseconds" INT(11) DEFAULT NULL, -- example: [1538426, 1539856, 1545928]
  "fastest_lap" INT(11) DEFAULT NULL, -- example: [14, 17, 16]
  "fastest_lap_time" VARCHAR(255) DEFAULT NULL, -- example: ['1:30.013', '1:29.937', '1:29.958']
  "fastest_lap_speed" VARCHAR(255) DEFAULT NULL, -- example: NULL
  "status_id" INT(11) NOT NULL DEFAULT '0', -- example: [1, 76, 3]
  PRIMARY KEY ("result_id")
)

CREATE TABLE "short_grand_prix_names" (
  "full_name" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['70th Anniversary Grand Prix', 'Abu Dhabi Grand Prix', 'Australian Grand Prix']
  "short_name" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Australia', 'Malaysia', 'China']
  PRIMARY KEY ("full_name")
)

CREATE TABLE "short_constructor_names" (
  "constructor_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['alphatauri', 'alpine', 'brabham-alfa_romeo']
  "short_name" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['Alpha Tauri', 'Alpine', 'Brabham']
  PRIMARY KEY ("constructor_ref")
)

CREATE TABLE "liveries" (
  "constructor_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['alfa', 'alphatauri', 'alpine']
  "start_year" INT(11) NOT NULL DEFAULT '0', -- example: [2019, 2020, 2021]
  "end_year" INT(11) NULL DEFAULT '0', -- example: [2002, 2005, 1991]
  "primary_hex_code" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['#900000', '#000000', '#F7A7D1']
  PRIMARY KEY ("constructor_ref", "start_year", "end_year")
)

CREATE TABLE "tdr_overrides" (
  "year" INT(11) NOT NULL DEFAULT '0', -- example: [2004, 2007, 2008]
  "constructor_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['toyota', 'mclaren', 'renault']
  "driver_ref" VARCHAR(255) NOT NULL DEFAULT '', -- example: ['matta', 'panis', 'trulli']
  "team_driver_rank" INT(11) NULL DEFAULT '0', -- example: [1, 2, 3]
  PRIMARY KEY ("year", "constructor_ref", "driver_ref")
)

CREATE TABLE circuits_ext(
  circuit_id INT, -- example: [1, 2, 3]
  circuit_ref TEXT, -- example: ['albert_park', 'sepang', 'bahrain']
  name TEXT, -- example: ['Albert Park Grand Prix Circuit', 'Sepang International Circuit', 'Bahrain International Circuit']
  location TEXT, -- example: ['Melbourne', 'Kuala Lumpur', 'Sakhir']
  country TEXT, -- example: ['Australia', 'Malaysia', 'Bahrain']
  lat REAL, -- example: [-37.8497, 2.76083, 26.0325]
  lng REAL, -- example: [144.968, 101.738, 50.5106]
  alt INT, -- example: [10, 18, 7]
  url TEXT, -- example: ['http://en.wikipedia.org/wiki/Melbourne_Grand_Prix_Circuit', 'http://en.wikipedia.org/wiki/Sepang_International_Circuit', 'http://en.wikipedia.org/wiki/Bahrain_International_Circuit']
  last_race_year, -- example: [2024, 2017, 2021]
  number_of_races -- example: [27, 19, 21]
)

CREATE TABLE constructors_ext(
  constructor_id INT, -- example: [1, 2, 3]
  constructor_ref TEXT, -- example: ['mclaren', 'bmw_sauber', 'williams']
  name TEXT, -- example: ['McLaren', 'BMW Sauber', 'Williams']
  nationality TEXT, -- example: ['British', 'German', 'French']
  url TEXT, -- example: ['http://en.wikipedia.org/wiki/McLaren', 'http://en.wikipedia.org/wiki/BMW_Sauber', 'http://en.wikipedia.org/wiki/Williams_Grand_Prix_Engineering']
  short_name -- example: ['McLaren', 'BMW Sauber', 'Williams']
)

CREATE TABLE drivers_ext(
  driver_id INT, -- example: [1, 2, 3]
  driver_ref TEXT, -- example: ['hamilton', 'heidfeld', 'rosberg']
  number INT, -- example: [44, 6, 14]
  code, -- example: ['HAM', 'HEI', 'ROS']
  forename TEXT, -- example: ['Lewis', 'Nick', 'Nico']
  surname TEXT, -- example: ['Hamilton', 'Heidfeld', 'Rosberg']
  full_name TEXT, -- example: ['Lewis Hamilton', 'Nick Heidfeld', 'Nico Rosberg']
  dob NUM, -- example: ['1985-01-07', '1977-05-10', '1985-06-27']
  nationality TEXT, -- example: ['British', 'German', 'Spanish']
  url TEXT -- example: ['http://en.wikipedia.org/wiki/Lewis_Hamilton', 'http://en.wikipedia.org/wiki/Nick_Heidfeld', 'http://en.wikipedia.org/wiki/Nico_Rosberg']
)

CREATE TABLE driver_standings_ext(
  driver_standings_id INT, -- example: [1, 2, 3]
  race_id INT, -- example: [18, 19, 20]
  driver_id INT, -- example: [1, 2, 3]
  points REAL, -- example: [10.0, 8.0, 6.0]
  position INT, -- example: [1, 2, 3]
  position_text TEXT, -- example: ['1', '2', '3']
  wins INT -- example: [1, 0, 2]
)

CREATE TABLE lap_times_ext(
  race_id INT, -- example: [1, 2, 3]
  driver_id INT, -- example: [1, 2, 3]
  lap INT, -- example: [1, 2, 3]
  position INT, -- example: [13, 12, 11]
  time TEXT, -- example: ['1:49.088', '1:33.740', '1:31.600']
  milliseconds INT, -- example: [109088, 93740, 91600]
  seconds REAL, -- example: [109.088, 93.74, 91.6]
  running_milliseconds -- example: [109088, 202828, 294428]
)

CREATE TABLE lap_time_stats(
  race_id INT, -- example: [1, 2, 3]
  driver_id INT, -- example: [1, 2, 3]
  avg_milliseconds, -- example: [97563.75862068965, 97635.6724137931, 97612.1724137931]
  avg_seconds, -- example: [97.56375862068965, 97.63567241379309, 97.61217241379309]
  stdev_milliseconds, -- example: [15927.054702406851, 14152.06249911631, 16170.377456130136]
  stdev_seconds -- example: [15.927054702406851, 14.152062499116306, 16.170377456130137]
)

CREATE TABLE races_ext(
  race_id INT, -- example: [1, 2, 3]
  year INT, -- example: [2009, 2008, 2007]
  round INT, -- example: [1, 2, 3]
  circuit_id INT, -- example: [1, 2, 17]
  name TEXT, -- example: ['Australian Grand Prix', 'Malaysian Grand Prix', 'Chinese Grand Prix']
  date NUM, -- example: ['2009-03-29', '2009-04-05', '2009-04-19']
  time NUM, -- example: ['06:00:00', '09:00:00', '07:00:00']
  url TEXT, -- example: ['http://en.wikipedia.org/wiki/2009_Australian_Grand_Prix', 'http://en.wikipedia.org/wiki/2009_Malaysian_Grand_Prix', 'http://en.wikipedia.org/wiki/2009_Chinese_Grand_Prix']
  fp1_date TEXT, -- example: ['2021-11-19', '2021-03-26', '2021-04-16']
  fp1_time TEXT, -- example: ['12:00:00', '14:00:00', '03:00:00']
  fp2_date TEXT, -- example: ['2021-11-19', '2021-03-26', '2021-04-16']
  fp2_time TEXT, -- example: ['15:00:00', '17:00:00', '06:00:00']
  fp3_date TEXT, -- example: ['2021-11-20', '2021-03-27', '2021-04-17']
  fp3_time TEXT, -- example: ['12:00:00', '14:00:00', '03:00:00']
  quali_date TEXT, -- example: ['2021-11-20', '2021-03-27', '2021-04-17']
  quali_time TEXT, -- example: ['15:00:00', '17:00:00', '06:00:00']
  sprint_date TEXT, -- example: ['2021-07-17', '2021-09-11', '2021-11-13']
  sprint_time TEXT, -- example: ['14:30:00', '19:30:00', '13:30:00']
  is_pit_data_available, -- example: [0, 1]
  short_name, -- example: ['Australia', 'Malaysia', 'China']
  has_sprint, -- example: [0, 1]
  max_points -- example: [10, 9, 25]
)

CREATE TABLE team_driver_ranks(
  year INT, -- example: [1950, 1951, 1952]
  constructor_id INT, -- example: [6, 51, 87]
  constructor_ref TEXT, -- example: ['ferrari', 'alfa', 'cooper']
  driver_id INT, -- example: [647, 687, 802]
  driver_ref TEXT, -- example: ['ascari', 'whitehead', 'serafini']
  team_driver_rank -- example: [1, 2, 3]
)

CREATE TABLE drives(
  year INT, -- example: [1950, 1951, 1952]
  driver_id INT, -- example: [427, 498, 501]
  drive_id, -- example: [1, 2, 3]
  constructor_id INT, -- example: [141, 105, 87]
  first_round INT, -- example: [2, 4, 3]
  last_round INT, -- example: [7, 6, 2]
  is_first_drive_of_season, -- example: [1, 0]
  is_final_drive_of_season -- example: [1, 0]
)

CREATE TABLE retirements(
  race_id INT, -- example: [18, 19, 20]
  driver_id INT, -- example: [7, 8, 9]
  lap, -- example: [56, 54, 48]
  position_order INT, -- example: [7, 8, 9]
  status_id INT, -- example: [5, 4, 3]
  retirement_type -- example: ['Retirement (Mechanical Problem)', 'Retirement (Driver Error)', 'Retirement (Disqualification)']
)

CREATE TABLE lap_positions(
  race_id INT, -- example: [1, 2, 3]
  driver_id INT, -- example: [1, 2, 3]
  lap INT, -- example: [0, 1, 2]
  position INT, -- example: [18, 13, 12]
  lap_type -- example: ['Starting Position - Grid Drop', 'Race', 'Retirement (Disqualification)']
)

