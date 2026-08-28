CREATE TABLE CountryLanguage (
    `CountryCode` text, -- Country Code. The unique code for country.
    `Language` text, -- Country language.
    `IsOfficial` text, -- Is Official. Information on whether the language is official in a given country. T / F.
    `Percentage` real, -- Percentage of language use.
    PRIMARY KEY (CountryCode,Language),
    FOREIGN KEY (CountryCode) REFERENCES Country (Code)
);

CREATE TABLE City (
    `ID` integer, -- the unique id for the city.
    `Name` text, -- the name of the city.
    `CountryCode` text, -- Country Code. the country code of the country that the city is belonged.
    `District` text, -- the district where the city locates.
    `Population` integer, -- the number of the population in the area. more population --> more crowded.
    primary key (ID),
    FOREIGN KEY (CountryCode) REFERENCES Country (Code)
);

CREATE TABLE Country (
    `Code` text, -- the unique country code of the country.
    `Name` text, -- the country name.
    `Continent` text, -- the continent that the country locates.
    `Region` text, -- the region that the country locates.
    `SurfaceArea` real, -- Surface Area. the surface area of the country.
    `IndepYear` integer, -- Independence Year. the year that the country declared independence.
    `Population` integer, -- the number of the population in the area. more population --> more crowded.
    `LifeExpectancy` real, -- Life Expectancy. the life expectancy at birth of the country. Life expectancy at birth is defined as how long, on average, a newborn can expect to live if current death rates do not change.
    `GNP` real, -- Gross National Product. the GNP of the country. GNP measures the total monetary value of the output produced by a country's residents.
    `GNPOld` real, -- Gross National Product Old. Gross national product - old value.
    `LocalName` text, -- Local Name. The country's local name.
    `GovernmentForm` text, -- Government Form. The country's goverment form. Republic: governmentform contains "Republic".
    `HeadOfState` text, -- Head Of State. The head of state full name.
    `Capital` integer, -- The country's capital. if the capital is null, it means this country doesn't have a city where a region's government is located.
    `Code2` text, -- The second country code.
    PRIMARY KEY (Code)
);

