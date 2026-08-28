CREATE TABLE Sales Orders (
    `OrderNumber` text, -- Order Number. the unique number identifying the order.
    `Sales Channel` text, -- Sales Channel.  In-Store.  Online.  Distributor.  Wholesale.
    `WarehouseCode` text, -- Warehouse Code. if the warehouse code is the same, it means this product is shipped from the same ware house.
    `ProcuredDate` text, -- Procured Date. date: month/date/year, 17--> 2017.
    `OrderDate` text, -- Order Date.
    `ShipDate` text, -- Ship Date.
    `DeliveryDate` text, -- Delivery Date. the difference "DeliveryDate - OrderDate" is smaller, it means faster delivery.
    `CurrencyCode` text, -- Currency Code. USD.
    `_SalesTeamID` integer, -- _Sales Team ID. Sales Team ID.
    `_CustomerID` integer, -- _Customer ID.
    `_StoreID` integer, -- _Store ID.
    `_ProductID` integer, -- _Product ID.
    `Order Quantity` integer, -- Order Quantity. 1 - 8:. higher means order quantity is higher or customer needs more.
    `Discount Applied` real, -- Discount Applied. 0.2: 20% discount.
    `Unit Price` text, -- No description.
    `Unit Cost` text, -- net profit = Unit Price - Unit Cost.
    primary key (OrderNumber)
);

CREATE TABLE Regions (
    `StateCode` text, -- state code. the unique code identifying the state.
    `State` text, -- full state name.
    `Region` text, -- the region where the state is located in.
    primary key (StateCode)
);

CREATE TABLE Sales Team (
    `SalesTeamID` integer, -- SalesTeam ID. unique sales team id.
    `Sales Team` text, -- sales team names.
    `Region` text, -- the region where the state is located in.
    primary key (SalesTeamID)
);

CREATE TABLE Store Locations (
    `StoreID` integer, -- Store ID. unique store id.
    `City Name` text, -- No description.
    `County` text, -- full county name.
    `StateCode` text, -- State Code. state code.
    `State` text, -- full state name.
    `Type` text, -- type of the store. City. Town. CDP (customer data platform). Unified Government. Consolidated Government. Other. Township. Urban County. Borough. Metropolitan Government.
    `Latitude` real, -- Latitude.
    `Longitude` real, -- Longitude. coordinates or detailed position: (Latitude, Longitude).
    `AreaCode` integer, -- Area Code.
    `Population` integer, -- Population.
    `Household Income` integer, -- Household Income.
    `Median Income` integer, -- Median Income.
    `Land Area` integer, -- Land Area.
    `Water Area` integer, -- Water Area.
    `Time Zone` text, -- Time Zone.
    primary key (StoreID)
);

CREATE TABLE Products (
    `ProductID` integer, -- product id. unique id number representing each product.
    `Product Name` text, -- product name. name of the product.
    primary key (ProductID)
);

CREATE TABLE Customers (
    `CustomerID` integer, -- customer id. unique id number indicating which customer.
    `Customer Names` text, -- customer names. the name of the customer.
    primary key (CustomerID)
);

