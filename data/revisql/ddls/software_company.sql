CREATE TABLE mailings3 (
    `REFID` integer, -- REFERENCE ID. unique id number identifying the customer.
    `REF_DATE` datetime, -- REFERENCE DATE. indicating the date when the mailing was sent.
    `RESPONSE` text, -- Actual response to the marketing incentive email. • True. • False. 1. any person who has not responded to a mailing within two months is considered to have responded negatively. 2. true respond to the mailing, otherwise, no.
    primary key (REFID)
);

CREATE TABLE Demog (
    `GEOID` integer, -- GEOGRAPHIC ID. unique geographic identifier.
    `INHABITANTS_K` real, -- INHABITANTS (THOUSANDS). number of inhabitants. the unit is K (thousands).
    `INCOME_K` real, -- INCOME (THOUSANDS). average income per inhabitant per month. the unit is dollar, it indicates the average income per inhabitant per month. some computation like: total income per year = INHABITANTS_K x INCOME_K x 12.
    `A_VAR1` real, -- No description.
    `A_VAR2` real, -- No description.
    `A_VAR3` real, -- No description.
    `A_VAR4` real, -- No description.
    `A_VAR5` real, -- No description.
    `A_VAR6` real, -- No description.
    `A_VAR7` real, -- No description.
    `A_VAR8` real, -- No description.
    `A_VAR9` real, -- No description.
    `A_VAR10` real, -- No description.
    `A_VAR11` real, -- No description.
    `A_VAR12` real, -- No description.
    `A_VAR13` real, -- No description.
    `A_VAR14` real, -- No description.
    `A_VAR15` real, -- No description.
    `A_VAR16` real, -- No description.
    `A_VAR17` real, -- No description.
    `A_VAR18` real, -- No description.
    primary key (GEOID)
);

CREATE TABLE Mailings1_2 (
    `REFID` integer, -- REFERENCE ID. unique id number identifying the customer.
    `REF_DATE` datetime, -- REFERENCE DATE. indicating the date when the mailing was sent.
    `RESPONSE` text, -- Response to the incentive mailing that marketing department sent. • True. • False. 1. any person who has not responded to a mailing within two months is considered to have responded negatively. 2. true respond to the mailing, otherwise, no.
    primary key (REFID)
);

CREATE TABLE Sales (
    `EVENTID` integer, -- EVENT ID. unique id of event (sales).
    `REFID` integer, -- REFERENCE ID. Reference to customer ID.
    `EVENT_DATE` datetime, -- EVENT DATE. date of sales.
    `AMOUNT` real, -- amount of sales.
    primary key (EVENTID)
);

CREATE TABLE Customers (
    `ID` integer, -- the unique number identifying the customer.
    `SEX` text, -- the sex of the customer.
    `MARITAL_STATUS` text, -- MARITAL STATUS. ï¿½ Never-married. ï¿½ Married-civ-spouse. ï¿½ Divorced. ï¿½ Widowed. ï¿½ Other. "Married-civ-spouse", "Divorced", "Widowed" mean customer has been married.
    `GEOID` integer, -- GEOGRAPHIC ID. geographic identifier.
    `EDUCATIONNUM` integer, -- EDUCATION NUMBER. the level of education. higher education number refers to higher education.
    `OCCUPATION` text, -- occupation of customers.
    `age` integer, -- age of customers. ï¿½ teenager: 13-19 years old. ï¿½ elder: people aged over 65.
    primary key (ID)
);

