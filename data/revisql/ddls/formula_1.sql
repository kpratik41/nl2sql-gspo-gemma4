CREATE TABLE circuits (
    `circuitId` integer , -- circuit Id. unique identification number of the circuit.
    `circuitRef` text, -- circuit reference name.
    `name` text, -- full name of circuit.
    `location` text, -- location of circuit.
    `country` text, -- country of circuit.
    `lat` real, -- latitude. latitude of location of circuit.
    `lng` real, -- longitude. longitude of location of circuit. Location coordinates: (lat, lng).
    `alt` integer , -- not useful.
    `url` text, -- url.
    primary key (circuitId)
);

CREATE TABLE status (
    `statusId` integer, -- status ID. the unique identification number identifying status.
    `status` text, -- full name of status.
    primary key (statusId)
);

CREATE TABLE drivers (
    `driverId` integer, -- driver ID. the unique identification number identifying each driver.
    `driverRef` text, -- driver reference name.
    `number` integer, -- number.
    `code` text, -- abbreviated code for drivers. if "null" or empty, it means it doesn't have code.
    `forename` text, -- forename.
    `surname` text, -- surname.
    `dob` date, -- date of birth.
    `nationality` text, -- nationality of drivers.
    `url` text, -- the introduction website of the drivers.
    primary key (driverId)
);

CREATE TABLE driverStandings (
    `driverStandingsId` integer, -- driver Standings Id. the unique identification number identifying driver standing records.
    `raceId` integer, -- constructor Reference name. id number identifying which races.
    `driverId` integer, -- id number identifying which drivers.
    `points` real, -- how many points acquired in each race.
    `position` integer, -- position or track of circuits.
    `wins` integer, -- wins.
    `positionText` text, -- position text. same with position, not quite useful.
    primary key (driverStandingsId),
    foreign key (raceId) references races(raceId),
    foreign key (driverId) references drivers(driverId)
);

CREATE TABLE races (
    `raceId` integer , -- race ID. the unique identification number identifying the race.
    `year` integer , -- year.
    `round` integer , -- round.
    `circuitId` integer , -- Circuit Id. circuit Id.
    `name` text, -- name of the race.
    `date` date, -- duration time.
    `time` text, -- time of the location.
    `url` text, -- introduction of races.
    primary key (raceId),
    foreign key (year) references seasons(year),
    foreign key (circuitId) references circuits(circuitId)
);

CREATE TABLE constructors (
    `constructorId` integer , -- constructor Id. the unique identification number identifying constructors.
    `constructorRef` text, -- Constructor Reference name.
    `name` text, -- full name of the constructor.
    `nationality` text, -- nationality of the constructor.
    `url` text, -- the introduction website of the constructor. How to find out the detailed introduction of the constructor: through its url.
    primary key (constructorId)
);

CREATE TABLE constructorResults (
    `constructorResultsId` integer, -- constructor Results Id.
    `raceId` integer, -- race Id. race id.
    `constructorId` integer, -- constructor Id. constructor id.
    `points` real, -- points.
    `status` text, -- status.
    primary key (constructorResultsId),
    foreign key (raceId) references races(raceId),
    foreign key (constructorId) references constructors(constructorId)
);

CREATE TABLE lapTimes (
    `raceId` integer , -- race ID. the identification number identifying race.
    `driverId` integer , -- driver ID. the identification number identifying each driver.
    `lap` integer , -- lap number.
    `position` integer , -- position or track of circuits.
    `time` text, -- lap time. in minutes / seconds / ...
    `milliseconds` integer , -- milliseconds.
    primary key (raceId, driverId, lap),
    foreign key (raceId) references races(raceId),
    foreign key (driverId) references drivers(driverId)
);

CREATE TABLE qualifying (
    `qualifyId` integer , -- qualify Id. the unique identification number identifying qualifying. How does F1 Sprint qualifying work? Sprint qualifying is essentially a short-form Grand Prix � a race that is one-third the number of laps of the main event on Sunday. However, the drivers are battling for positions on the grid for the start of Sunday's race.
    `raceId` integer , -- race Id. the identification number identifying each race.
    `driverId` integer , -- driver Id. the identification number identifying each driver.
    `constructorId` integer , -- constructor id. constructor Id.
    `number` integer , -- number.
    `position` integer , -- position or track of circuit.
    `q1` text, -- qualifying 1. time in qualifying 1. in minutes / seconds / ... Q1 lap times determine pole position and the order of the front 10 positions on the grid. The slowest driver in Q1 starts 10th, the next starts ninth and so on. All 20 F1 drivers participate in the first period, called Q1, with each trying to set the fastest time possible. Those in the top 15 move on to the next period of qualifying, called Q2. The five slowest drivers are eliminated and will start the race in the last five positions on the grid.
    `q2` text, -- qualifying 2. time in qualifying 2. in minutes / seconds / ... only top 15 in the q1 has the record of q2. Q2 is slightly shorter but follows the same format. Drivers try to put down their best times to move on to Q1 as one of the 10 fastest cars. The five outside of the top 10 are eliminated and start the race from 11th to 15th based on their best lap time.
    `q3` text, -- qualifying 3. time in qualifying 3. in minutes / seconds / ... only top 10 in the q2 has the record of q3.
    primary key (qualifyId),
    foreign key (raceId) references races(raceId),
    foreign key (driverId) references drivers(driverId),
    foreign key (constructorId) references constructors(constructorId)
);

CREATE TABLE pitStops (
    `raceId` integer , -- race ID. the identification number identifying race.
    `driverId` integer , -- driver ID. the identification number identifying each driver.
    `stop` integer , -- stop number.
    `lap` integer , -- lap number.
    `time` text, -- time. exact time.
    `duration` text, -- duration time. seconds/.
    `milliseconds` integer , -- milliseconds.
    primary key (raceId, driverId, stop),
    foreign key (raceId) references races(raceId),
    foreign key (driverId) references drivers(driverId)
);

CREATE TABLE seasons (
    `year` integer, -- race ID. the unique identification number identifying the race.
    `url` text, -- website link of season race introduction.
    primary key (year)
);

CREATE TABLE constructorStandings (
    `constructorStandingsId` integer , -- constructor Standings Id. unique identification of the constructor standing records.
    `raceId` integer , -- race id. id number identifying which races.
    `constructorId` integer , -- constructor id. id number identifying which id.
    `points` real, -- how many points acquired in each race.
    `position` integer , -- position or track of circuits.
    `positionText` text, -- position text. same with position, not quite useful.
    `wins` integer , -- wins.
    primary key (constructorStandingsId),
    foreign key (raceId) references races(raceId),
    foreign key (constructorId) references constructors(constructorId)
);

CREATE TABLE results (
    `resultId` integer, -- Result ID. the unique identification number identifying race result.
    `raceId` integer , -- race ID. the identification number identifying the race.
    `driverId` integer , -- driver ID. the identification number identifying the driver.
    `constructorId` integer , -- constructor Id. the identification number identifying which constructors.
    `number` integer , -- number.
    `grid` integer , -- the number identifying the area where cars are set into a grid formation in order to start the race.
    `position` integer , -- The finishing position or track of circuits.
    `positionText` text, -- position text. not quite useful.
    `positionOrder` integer , -- position order. the finishing order of positions.
    `points` real, -- points.
    `laps` integer , -- lap number.
    `time` text, -- finish time. 1. if the value exists, it means the driver finished the race. 2. Only the time of the champion shows in the format of "minutes: seconds.millionsecond", the time of the other drivers shows as "seconds.millionsecond" , which means their actual time is the time of the champion adding the value in this cell.
    `milliseconds` integer , -- the actual finishing time of drivers in milliseconds. the actual finishing time of drivers.
    `fastestLap` integer , -- fastest lap. fastest lap number.
    `rank` integer , -- starting rank positioned by fastest lap speed.
    `fastestLapTime` text, -- fastest Lap Time. faster (smaller in the value) "fastestLapTime" leads to higher rank (smaller is higher rank).
    `fastestLapSpeed` text, -- fastest Lap Speed. (km / h).
    `statusId` integer, -- status Id. status ID. its category description appear in the table status.
    primary key (resultId),
    foreign key (raceId) references races(raceId),
    foreign key (driverId) references drivers(driverId),
    foreign key (constructorId) references constructors(constructorId),
    foreign key (statusId) references status(statusId)
);

