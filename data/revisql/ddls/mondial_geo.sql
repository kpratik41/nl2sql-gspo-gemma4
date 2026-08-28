CREATE TABLE economy (
    `Country` text, -- the country code.
    `GDP` real, -- gross domestic product.
    `Agriculture` real, -- percentage of agriculture of the GDP.
    `Service` real, -- percentage of services of the GDP,.
    `Industry` real, -- percentage of industry of the GDP.
    `Inflation` real, -- inflation rate (per annum),.
    primary key (Country)
);

CREATE TABLE geo_island (
    `Island` text, -- the name of the island.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, Island),
    foreign key (Province, Country) references province
);

CREATE TABLE islandIn (
    `Island` text, -- No description.
    `Sea` text, -- No description.
    `Lake` text, -- No description.
    `River` text, -- No description.

);

CREATE TABLE ethnicGroup (
    `Country` text, -- the country code.
    `Name` text, -- name of the language.
    `Percentage` real, -- percentage of the language in this country. %.
    primary key (Name, Country)
);

CREATE TABLE island (
    `Name` text, -- the name of the island.
    `Islands` text, -- the group of islands where it belongs to.
    `Area` real, -- the area of the island.
    `Height` real, -- the maximal elevation of the island.
    `Type` text, -- the type of the island.
    `Longitude` real, -- Longitude.
    `Latitude` real, -- Latitude.
    primary key (Name)
);

CREATE TABLE sea (
    `Name` text, -- the name of the sea.
    `Depth` real, -- the maximal depth of the sea.
    primary key (Name)
);

CREATE TABLE river (
    `Name` text, -- the name of the river.
    `River` text, -- the river where it finally flows to.
    `Lake` text, -- the lake where it finally flows to.
    `Sea` text, -- the sea where it finally flows to. (note that at most one out of {river,lake,sea} can be non-null.
    `Length` real, -- the length of the river.
    `SourceLongitude` real, -- the longitude of its source.
    `SourceLatitude` real, -- the latitude of its source.
    `Mountains` text, -- the mountains where its source is located.
    `SourceAltitude` real, -- the elevation (above sea level) of its source.
    `EstuaryLongitude` real, -- the coordinates of its estuary.
    `EstuaryLatitude` real, -- the latitude of its estuary.
    primary key (Name)
);

CREATE TABLE organization (
    `Abbreviation` text, -- its abbreviation.
    `Name` text, -- the full name of the organization.
    `City` text, -- the city where the headquarters are located.
    `Country` text, -- the code of the country where the headquarters are located.
    `Province` text, -- the name of the province where the headquarters are located,.
    `Established` date, -- date of establishment.
    primary key (Abbreviation),
    foreign key (City, Province) references city
    foreign key (Province, Country) references province
);

CREATE TABLE geo_desert (
    `Desert` text, -- the name of the desert.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, Desert),
    foreign key (Province, Country) references province
);

CREATE TABLE geo_source (
    `River` text, -- the name of the river.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, River),
    foreign key (Province, Country) references province
);

CREATE TABLE located (
    `City` text, -- the name of the city.
    `Country` text, -- the country code where the city belongs to.
    `Province` text, -- the province where the city belongs to.
    `River` text, -- the river where it is located at.
    `Lake` text, -- the lake where it is located at.
    `Sea` text, -- the sea where it is located at. Note that for a given city, there can be several lakes/seas/rivers where it is located at.
    foreign key (City, Province) references city
    foreign key (Province, Country) references province
);

CREATE TABLE politics (
    `Country` text, -- the country code.
    `Independence` date, -- date of independence. Commonsense evidence:. if the value is null or empty, it means this country is not independent.
    `Dependent` text, -- the country code where the area belongs to.
    `Government` text, -- type of government.
    primary key (Country)
);

CREATE TABLE geo_mountain (
    `Mountain` text, -- the name of the mountain.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, Mountain),
    foreign key (Province, Country) references province
);

CREATE TABLE population (
    `Country` text, -- the country code.
    `Population_Growth` real, -- population growth. population growth rate. per annum.
    `Infant_Mortality` real, -- infant mortality. per thousand.
    primary key (Country)
);

CREATE TABLE geo_sea (
    `Sea` text, -- the name of the sea.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, Sea),
    foreign key (Province, Country) references province
);

CREATE TABLE mergesWith (
    `Sea1` text, -- the name of the mountain.
    `Sea2` text, -- the country code where it is located.
    primary key (Sea1, Sea2)
);

CREATE TABLE encompasses (
    `Country` text, -- a country code.
    `Continent` text, -- the continent name.
    `Percentage` real, -- how much of the area of a country belongs to the continent. %.
    primary key (Country, Continent)
);

CREATE TABLE province (
    `Name` text, -- the name of the administrative division.
    `Country` text, -- the country code where it belongs to.
    `Area` integer, -- the total area of the province,.
    `Population` real, -- the population of the province.
    `Capital` text, -- the name of the capital. if null, doesn't have capital.
    `CapProv` text, -- capital province. the name of the province where the capital belongs to. note that capprov is not necessarily equal to name.
    primary key (Name, Country)
);

CREATE TABLE continent (
    `Name` text, -- name of the continent.
    `Area` real, -- total area of the continent.
    primary key (Name)
);

CREATE TABLE country (
    `Name` text, -- the country name.
    `Code` text, -- country code.
    `Capital` text, -- the name of the capital,.
    `Province` text, -- the province where the capital belongs to,.
    `Area` real, -- the total area,.
    `Population` integer, -- the population number.
    primary key (Code)
);

CREATE TABLE geo_river (
    `River` text, -- the name of the river.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, River),
    foreign key (Province, Country) references province
);

CREATE TABLE target (
    `Country` text, -- No description.
    `Target` real, -- No description.
    primary key (Country)
);

CREATE TABLE religion (
    `Country` text, -- the country code.
    `Name` text, -- name of the religion.
    `Percentage` real, -- percentage of the language in this country. %.
    primary key (Name, Country)
);

CREATE TABLE geo_lake (
    `Lake` text, -- the name of the lake.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, Lake),
    foreign key (Province, Country) references province
);

CREATE TABLE desert (
    `Name` text, -- the name of the desert.
    `Area` real, -- the total area of the desert.
    `Longitude` real, -- Longitude.
    `Latitude` real, -- Latitude. coordinate: (Longitude, Latitude).
    primary key (Name)
);

CREATE TABLE city (
    `Name` text, -- name of city.
    `Country` text, -- the code of the country where it belongs to.
    `Province` text, -- the name of the province where it belongs to.
    `Population` integer, -- population of the city.
    `Longitude` real, -- geographic longitude.
    `Latitude` real, -- geographic latitude.
    primary key (Name, Province),
    foreign key (Province, Country) references province
);

CREATE TABLE isMember (
    `Country` text, -- No description.
    `Organization` text, -- No description.
    `Type` text, -- No description.
    primary key (Country, Organization)
);

CREATE TABLE borders (
    `Country1` text, -- a country code.
    `Country2` text, -- a country code.
    `Length` real, -- length of the border between country1 and country2.
    primary key (Country1, Country2)
);

CREATE TABLE locatedOn (
    `City` text, -- the name of the city.
    `Province` text, -- the province where the city belongs to.
    `Country` text, -- the country code where the city belongs to.
    `Island` text, -- the island it is (maybe only partially) located on.
    primary key (City, Province, Country, Island),
    foreign key (City, Province) references city
    foreign key (Province, Country) references province
);

CREATE TABLE language (
    `Country` text, -- the country code.
    `Name` text, -- name of the language.
    `Percentage` real, -- percentage of the language in this country. %.
    primary key (Name, Country)
);

CREATE TABLE mountain (
    `Name` text, -- the name of the mountain.
    `Mountains` text, -- the mountains where it belongs to.
    `Height` real, -- the maximal elevation of the summit of the mountain.
    `Type` text, -- the sea where it finally flows to. (note that at most one out of {river,lake,sea} can be non-null.
    `Longitude` real, -- the length of the river.
    `Latitude` real, -- the longitude of its source.
    primary key (Name)
);

CREATE TABLE lake (
    `Name` text, -- the name of the lake.
    `Area` real, -- the total area of the lake.
    `Depth` real, -- the depth of the lake.
    `Altitude` real, -- the altitude (above sea level) of the lake.
    `River` text, -- the river that flows out of the lake.
    `Type` text, -- the type of the lake.
    `Longitude` real, -- longitude of lake.
    `Latitude` real, -- latitude of lake.
    primary key (Name)
);

CREATE TABLE geo_estuary (
    `River` text, -- the name of the river.
    `Country` text, -- the country code where it is located.
    `Province` text, -- the province of this country.
    primary key (Province, Country, River),
    foreign key (Province, Country) references province
);

CREATE TABLE mountainOnIsland (
    `Mountain` text, -- the name of the mountain.
    `Island` text, -- the name of the island.
    primary key (Mountain, Island)
);

