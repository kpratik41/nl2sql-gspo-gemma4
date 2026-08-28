CREATE TABLE customers (
    `CustomerID` integer, -- identification of the customer.
    `Segment` text, -- client segment.
    `Currency` text, -- Currency.
    primary key (CustomerID)
);

CREATE TABLE gasstations (
    `GasStationID` integer, -- Gas Station ID.
    `ChainID` integer, -- Chain ID.
    `Country` text, -- No description.
    `Segment` text, -- chain segment.
    primary key (GasStationID)
);

CREATE TABLE products (
    `ProductID` integer, -- Product ID.
    `Description` text, -- Description.
    primary key (ProductID)
);

CREATE TABLE yearmonth (
    `CustomerID` integer, -- Customer ID.
    `Date` text, -- Date.
    `Consumption` real, -- consumption.
    primary key (Date, CustomerID)
);

CREATE TABLE transactions_1k (
    `TransactionID` integer, -- Transaction ID.
    `Date` date, -- Date.
    `Time` text, -- Time.
    `CustomerID` integer, -- Customer ID.
    `CardID` integer, -- Card ID.
    `GasStationID` integer, -- Gas Station ID.
    `ProductID` integer, -- Product ID.
    `Amount` integer, -- Amount.
    `Price` real, -- Price. total price = Amount x Price.
    primary key (TransactionID)
);

