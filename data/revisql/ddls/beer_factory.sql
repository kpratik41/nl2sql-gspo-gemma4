CREATE TABLE rootbeerreview (
    `CustomerID` integer, -- customer id. the id of the customer.
    `BrandID` integer, -- brand id. the id of the brand.
    `StarRating` integer, -- star rating. the star rating of the root beer. root beer with higher star rating has higher market evaluation and acceptance.
    `ReviewDate` date, -- review date. the review date. yyyy-mm-dd.
    `Review` text, -- the specific review content.
    primary key (CustomerID, BrandID),
    foreign key (CustomerID) references customers(CustomerID),
    foreign key (BrandID) references rootbeerbrand(BrandID)
);

CREATE TABLE geolocation (
    `LocationID` integer, -- location id. the id of the location.
    `Latitude` real, -- the latitude of the location.
    `Longitude` real, -- the longitude of the location. precise location / coordinate = POINT(latitude, longitude).
    primary key (LocationID),
    foreign key (LocationID) references location(LocationID)
);

CREATE TABLE customers (
    `CustomerID` integer, -- customer id. the unique id for the customer.
    `First` text, -- first name. the first name of the customer.
    `Last` text, -- last name. the last name of the customer.
    `StreetAddress` text, -- street address. the address of the customer.
    `City` text, -- the city where the customer lives.
    `State` text, -- the state code. please refer to https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971. and mention its corresponding state name in the question. i.e. New York-- NY.
    `ZipCode` integer, -- zip code. the zip code.
    `Email` text, -- the customer's email.
    `PhoneNumber` text, -- phone number. the customer's phone number.
    `FirstPurchaseDate` date, -- first purchase date. the first purchase date of the customer. yyyy-mm-dd.
    `SubscribedToEmailList` text, -- subscribed to email list. whether the customer subscribe to the email list. 'true' means the user permits the company to send regular emails to them.
    `Gender` text, -- the customer's gender.
    primary key (CustomerID)
);

CREATE TABLE location (
    `LocationID` integer, -- location id. the unique id for the location.
    `LocationName` text, -- location name. the name of the location.
    `StreetAddress` text, -- street address. the street address.
    `City` text, -- the city where the location locates.
    `State` text, -- the state code.
    `ZipCode` integer, -- zip code. the zip code of the location.
    primary key (LocationID),
    foreign key (LocationID) references geolocation(LocationID)
);

CREATE TABLE transaction (
    `TransactionID` integer, -- transaction id. the unique id for the transaction.
    `CreditCardNumber` integer, -- credit card number. the number of the credit card used for payment.
    `CustomerID` integer, -- customer id. the customer id.
    `TransactionDate` date, -- transaction date. the transaction date. yyyy-mm-dd.
    `CreditCardType` text, -- credit card type. the credit card type.
    `LocationID` integer, -- location id. the location id of the selling company.
    `RootBeerID` integer, -- root beer id. the root beer id.
    `PurchasePrice` real, -- purchase price. the unit purchase price of the root beer. us dollars.
    primary key (TransactionID),
    foreign key (CustomerID) references customers(CustomerID),
    foreign key (LocationID) references location(LocationID),
    foreign key (RootBeerID) references rootbeer(RootBeerID)
);

CREATE TABLE rootbeerbrand (
    `BrandID` integer, -- brand id. the unique id for the brand.
    `BrandName` text, -- brand name. the brand name.
    `FirstBrewedYear` integer, -- first brewed year. the first brewed year of the brand. brand with earlier first brewed year has a much longer brewed history.
    `BreweryName` text, -- brewery name. the brewery name.
    `City` text, -- the city where the brewery locates.
    `State` text, -- the state code.
    `Country` text, -- the country where the brewery locates. can find its corresponding continent. e.g., U.S.--> North America.
    `Description` text, -- the description of the brand.
    `CaneSugar` text, -- cane sugar. whether the drink contains cane sugar.
    `CornSyrup` text, -- corn syrup. whether the drink contains the corn syrup.
    `Honey` text, -- whether the drink contains the honey. if the beer has honey, it means this beer is sweeter or has sweetness.
    `ArtificialSweetener` text, -- artificial sweetener. whether the drink contains the artificial sweetener. if the beer has artificial sweetener, it means this beer is sweeter or has sweetness.
    `Caffeinated` text, -- whether the drink is caffeinated.
    `Alcoholic` text, -- whether the drink is alcoholic.
    `AvailableInCans` text, -- available in cans. whether the drink is available in cans.
    `AvailableInBottles` text, -- available in bottles. whether the drink is available in bottles.
    `AvailableInKegs` text, -- available in kegs. whether the drink is available in kegs.
    `Website` text, -- the website of the brand.
    `FacebookPage` text, -- facebook page. the facebook page of the brand. if not, it means this brand doesn't advertise on facebook.
    `Twitter` text, -- the twitter of the brand. if not, it means this brand doesn't advertise on twitter.
    `WholesaleCost` real, -- wholesale cost. the wholesale cost.
    `CurrentRetailPrice` real, -- current retail price. the current retail price. The unit profit available to wholesalers = current retail price - wholesale cost.
    primary key (BrandID)
);

CREATE TABLE rootbeer (
    `RootBeerID` integer, -- root beer id. the unique id for the root beer.
    `BrandID` integer, -- brandid. the brand id.
    `ContainerType` text, -- container type. the type of the container.
    `LocationID` integer, -- location id. the location id of the selling company.
    `PurchaseDate` date, -- purchase date. the purchase date of the root beer.
    primary key (RootBeerID),
    foreign key (LocationID) references geolocation(LocationID),
    foreign key (LocationID) references location(LocationID),
    foreign key (BrandID) references rootbeerbrand(BrandID)
);

