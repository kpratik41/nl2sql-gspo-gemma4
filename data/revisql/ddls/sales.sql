CREATE TABLE Products (
    `ProductID` integer, -- Product ID. the unique id of the product.
    `Name` text, -- the product name.
    `Price` real, -- the price of the product. unit price. if the price = 0.0, it means this product is free or a gift.
    primary key (ProductID)
);

CREATE TABLE Employees (
    `EmployeeID` integer, -- Employee ID. the unique id of the employee.
    `FirstName` text, -- First Name. the employee's first name.
    `MiddleInitial` text, -- Middle Initial. the employee's middle initial.
    `LastName` text, -- Last Name. the employee's last name.
    primary key (EmployeeID)
);

CREATE TABLE Sales (
    `SalesID` integer, -- Sales ID. the unique id of the sales.
    `SalesPersonID` integer, -- SalesPerson ID. the unique id of the sale person.
    `CustomerID` integer, -- Customer ID. the unique id of the customer.
    `ProductID` integer, -- Product ID. the unique id of the product.
    `Quantity` integer, -- trading quantity. total price = quantity x Products' Price.
    primary key (SalesID),
    foreign key (SalesPersonID) references Employees (EmployeeID)
    foreign key (CustomerID) references Customers (CustomerID)
    foreign key (ProductID) references Products (ProductID)
);

CREATE TABLE Customers (
    `CustomerID` integer, -- Customer ID. the unique id of the customer.
    `FirstName` text, -- First Name. the customer's first name.
    `MiddleInitial` text, -- Middle Initial. the customer's middle initial.
    `LastName` text, -- Last Name. the customer's last name.
    primary key (CustomerID)
);

