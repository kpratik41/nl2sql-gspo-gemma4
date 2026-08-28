CREATE TABLE data (
    `ID` integer, -- unique ID for each car.
    `mpg` real, -- mileage per gallon. mileage of the car in miles per gallon. The car with higher mileage is more fuel-efficient.
    `cylinders` integer, -- number of cylinders. the number of cylinders present in the car.
    `displacement` real, -- engine displacement in cubic mm. sweep volume = displacement / no_of cylinders.
    `horsepower` integer, -- horse power. horse power associated with the car. horse power is the metric used to indicate the power produced by a car's engine - the higher the number, the more power is sent to the wheels and, in theory, the faster it will go.
    `weight` integer, -- weight of the car in lbs. A bigger, heavier vehicle provides better crash protection than a smaller.
    `acceleration` real, -- acceleration of the car in miles per squared hour.
    `model` integer, -- the year when the car model was introduced in the market. 0 --> 1970.
    `car_name` text, -- car name. name of the car.
    primary key (ID),
    foreign key (ID) references price(ID)
);

CREATE TABLE price (
    `ID` integer, -- unique ID for each car.
    `price` real, -- price of the car in USD.
    primary key (ID)
);

CREATE TABLE production (
    `ID` integer, -- the id of the car.
    `model_year` integer, -- model year. year when the car model was introduced in the market.
    `country` integer, -- country id to which the car belongs. Japan --> Asia. USA --> North America.
    primary key (ID, model_year),
    foreign key (country) references country(origin),
    foreign key (ID) references data(ID),
    foreign key (ID) references price(ID)
);

CREATE TABLE country (
    `origin` integer, -- the unique identifier for the origin country.
    `country` text, -- the origin country of the car.
    primary key (origin)
);

