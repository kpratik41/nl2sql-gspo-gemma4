CREATE TABLE avoid (
    `zip_code` integer, -- zip code. the zip code of the bad alias.
    `bad_alias` text, -- bad alias. the bad alias.
    primary key (zip_code, bad_alias),
    foreign key (zip_code) references zip_data(zip_code)
);

CREATE TABLE area_code (
    `zip_code` integer, -- zip code. the zip code of the area.
    `area_code` integer, -- area code. the code of the area.
    primary key (zip_code, area_code),
    foreign key (zip_code) references zip_data(zip_code)
);

CREATE TABLE CBSA (
    `CBSA` integer, -- the code of the cbsa officer.
    `CBSA_name` text, -- cbsa name. the name and the position of the cbsa officer.
    `CBSA_type` text, -- cbsa type. the office type of the officer.
    primary key (CBSA)
);

CREATE TABLE zip_congress (
    `zip_code` integer, -- zip code. the zip code of the district.
    `district` text, -- the district.
    primary key (zip_code, district),
    foreign key (district) references congress(cognress_rep_id),
    foreign key (zip_code) references zip_data(zip_code)
);

CREATE TABLE zip_data (
    `zip_code` integer, -- zip code. the zip code of the postal point.
    `city` text, -- the city where the postal point locates.
    `state` text, -- the state of the city.
    `multi_county` text, -- multi county. whether the county that the city belongs to is multi_county. The meaning of multi_county is consisting of or involving multiple countries.
    `type` text, -- the type of the postal point.
    `organization` text, -- the organization to which the postal point belongs. 'No data' means the postal point is not affiliated with any organization.
    `time_zone` text, -- time zone. the time zone of the postal point location.
    `daylight_savings` text, -- daylight savings. whether the location implements daylight savings. Daylight saving is the practice of advancing clocks (typically by one hour) during warmer months so that darkness falls at a later clock time. As a result, there is one 23-hour day in late winter or early spring and one 25-hour day in autumn.
    `latitude` real, -- the latitude of the postal point location.
    `longitude` real, -- the longitude of the postal point location.
    `elevation` integer, -- the elevation of the postal point location.
    `state_fips` integer, -- state fips. state-level FIPS code.
    `county_fips` integer, -- county fips. county-level FIPS code. FIPS codes are numbers that uniquely identify geographic areas. The number of digits in FIPS codes varies depending on the level of geography.
    `region` text, -- the region where the postal point locates.
    `division` text, -- the division of the organization.
    `population_2020` integer, -- population 2020. the population of the residential area in 2020. 'No data' means that the postal point is not affiliated with any organization. Or the organization has no division.
    `population_2010` integer, -- population 2010. the population of the residential area in 2010.
    `households` integer, -- the number of the households in the residential area.
    `avg_house_value` integer, -- average house value. the average house value in the residential area.
    `avg_income_per_household` integer, -- average income per household. the average income per household in the residential area.
    `persons_per_household` real, -- persons per household. the number of persons per household residential area.
    `white_population` integer, -- white population. the population of white people in the residential area.
    `black_population` integer, -- black population. the population of black people in the residential area.
    `hispanic_population` integer, -- Hispanic population. the population of Hispanic people in the residential area.
    `asian_population` integer, -- Asian population. the population of Asian people in the residential area.
    `american_indian_population` integer, -- American Indian population. the population of American Indian people in the residential area.
    `hawaiian_population` integer, -- Hawaiian population. the population of Hawaiian people in the residential area.
    `other_population` integer, -- other population. the population of other races in the residential area.
    `male_population` integer, -- male population. the population of males in the residential area.
    `female_population` integer, -- female population. the population of females in the residential area.
    `median_age` real, -- median age. the median age of the residential area. gender ratio in the residential area = male_population / female_population if female_population != 0.
    `male_median_age` real, -- male median age. the male median age in the residential area.
    `female_median_age` real, -- female median age. the female median age in the residential area.
    `residential_mailboxes` integer, -- residential mailboxes. the number of residential mailboxes in the residential area.
    `business_mailboxes` integer, -- business mailboxes. the number of business mailboxes in the residential area.
    `total_delivery_receptacles` integer, -- total delivery receptacles. the total number of delivery receptacles.
    `businesses` integer, -- the number of businesses in the residential area.
    `1st_quarter_payroll` integer, -- 1st quarter payroll. the total wages reported in the 1st quarter payroll report of the residential area.
    `annual_payroll` integer, -- annual payroll. the total wages reported in the annual payroll report of the residential area.
    `employees` integer, -- the number of employees in the residential area. Employers must submit periodic payroll reports to the Internal Revenue Service and state taxing authority. The report is used by employers to report wage information.
    `water_area` real, -- water area. the water area of the residential area.
    `land_area` real, -- land area. the land area of the residential area.
    `single_family_delivery_units` integer, -- single-family delivery units. the number of single-family delivery units in the residential area.
    `multi_family_delivery_units` integer, -- multi-family delivery units. the number of single-family delivery units in the residential area. A single-family unit can be described as a single unit that accommodates only one family or tenant at a time on rent.
    `total_beneficiaries` integer, -- total beneficiaries. the total number of beneficiaries of the postal service. multi-family units are usually under a greater surface area with the capacity to accommodate more than one tenant or multiple families on rent at the same time.
    `retired_workers` integer, -- retired workers. the number of retired workers in the residential area.
    `disabled_workers` integer, -- disable workers. the number of disabled workers in the residential area.
    `parents_and_widowed` integer, -- parents and widowed. the number of parents and widowed in the residential area.
    `spouses` integer, -- the number of spouses in the residential area.
    `children` integer, -- the number of children in the residential area.
    `over_65` integer, -- over 65. the number of people whose age is over 65 in the residential area.
    `monthly_benefits_all` integer, -- monthly benefits all. all benefit payments by month. no. of over_65 can refer to elders. Higher means more elders. if the ratio: over_65 / children is higher: the aging is more serious.
    `monthly_benefits_retired_workers` integer, -- monthly benefits retired workers. monthly benefit payments for retired workers. usd.
    `monthly_benefits_widowed` integer, -- monthly benefits widowed. monthly benefit payments for retired workers.
    `CBSA` integer, -- the code of the cbsa officer.
    primary key (zip_code),
    foreign key (state) references state(abbreviation),
    foreign key (CBSA) references CBSA(CBSA)
);

CREATE TABLE state (
    `abbreviation` text, -- the abbreviation of the state name.
    `name` text, -- the state name.
    primary key (abbreviation)
);

CREATE TABLE congress (
    `cognress_rep_id` text, -- congress representative id. the representative id of congress representatives.
    `first_name` text, -- first name. the first name of the congress representative.
    `last_name` text, -- last name. the last name of the congress representative.
    `CID` text, -- the unique identifier for the congress representative.
    `party` text, -- the party of the representative.
    `state` text, -- the state that the representative is from.
    `abbreviation` text, -- the abbreviation of the state.
    `House` text, -- the house that the representative is from.
    `District` integer, -- the id of the district that the representative represents. The state is divided into different districts. The districts under the same state will be numbered uniformly. 'NA' means that the state is not divided into districts.
    `land_area` real, -- land area. the land area of the district.
    primary key (cognress_rep_id),
    foreign key (abbreviation) references state(abbreviation)
);

CREATE TABLE country (
    `zip_code` integer, -- zip code. the zip code of the state in the country.
    `county` text, -- county. the county.
    `state` text, -- the state of the county.
    primary key (zip_code, county),
    foreign key (zip_code) references zip_data(zip_code),
    foreign key (state) references state(abbreviation)
);

CREATE TABLE alias (
    `zip_code` integer, -- zip code. the zip code of the alias.
    `alias` text, -- the alias of the city.
    primary key (zip_code),
    foreign key (zip_code) references zip_data(zip_code)
);

