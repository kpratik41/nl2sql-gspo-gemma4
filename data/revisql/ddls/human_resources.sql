CREATE TABLE employee (
    `ssn` text, -- social security number. employee's ssn.
    `lastname` text, -- last name. employee's last name.
    `firstname` text, -- first name. employee's first name. employee's full name is firstname lastname .
    `hiredate` text, -- hire date. employee's hire date. yyyy-mm-dd.
    `salary` text, -- employee's salary. the unit is dollar per year.
    `gender` text, -- employee's gender.
    `performance` text, -- employee's performance. Good / Average/ Poor.
    `positionID` integer, -- position id. the unique id for position.
    `locationID` integer, -- location id. the unique id for location.
    primary key (ssn),
    foreign key (locationID) references location(locationID),
    foreign key (positionID) references position(positionID)
);

CREATE TABLE location (
    `locationID` integer, -- location id. the unique id for location.
    `locationcity` text, -- location city. the location city.
    `address` text, -- the detailed address of the location.
    `state` text, -- the state abbreviation.
    `zipcode` integer, -- zip code. zip code of the location.
    `officephone` text, -- office phone. the office phone number of the location.
    primary key (locationID)
);

CREATE TABLE position (
    `positionID` integer, -- position id. the unique id for position.
    `positiontitle` text, -- position title. the position title.
    `educationrequired` text, -- education required. the required education year. Generally, more complex work requires more education year.
    `minsalary` text, -- minimum salary. minimum salary of this position.
    `maxsalary` text, -- maximum salary. maximum salary of this position.
    primary key (positionID)
);

