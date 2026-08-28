CREATE TABLE OrderDetails (
    `OrderID` integer, -- Order ID. the unique id for orders.
    `ProductID` integer, -- Product ID. the unique id for products.
    `Quantity` integer, -- the quantity of the ordered products.
    `OrderDetailID` text, -- order detail id. id of the order detail.
    primary key (OrderDetailID),
    FOREIGN KEY (OrderID) REFERENCES Orders (OrderID),
    FOREIGN KEY (ProductID) REFERENCES Products (ProductID)
);

CREATE TABLE Orders (
    `OrderID` integer, -- Order ID. the unique id for orders.
    `CustomerID` text, -- Customer ID. the unique id for customers.
    `EmployeeID` integer, -- Employee ID. the unique id for employees.
    `OrderDate` datetime, -- Order Date. the order date.
    `ShipperID` integer, -- Shipper ID. the unique id for shippers.
    primary key (OrderID),
    FOREIGN KEY (EmployeeID) REFERENCES Employees (EmployeeID),
    FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID),
    FOREIGN KEY (ShipperID) REFERENCES Shippers (ShipperID)
);

CREATE TABLE Products (
    `ProductID` integer, -- Product ID. the unique id for products.
    `ProductName` text, -- Product Name. the name of the product.
    `SupplierID` integer, -- Supplier ID. the unique id for supplier.
    `CategoryID` integer, -- Category ID. the unique id for the product category.
    `Price` real, -- the price.
    `Unit` text, -- Unit. the unit of the product.
    primary key (ProductID),
    FOREIGN KEY (CategoryID) REFERENCES Categories (CategoryID),
    FOREIGN KEY (SupplierID) REFERENCES Suppliers (SupplierID)
);

CREATE TABLE Employees (
    `EmployeeID` integer, -- Employee ID. the unique id for employees.
    `LastName` text, -- Last Name. the employee's last name.
    `FirstName` text, -- First Name. the employee's first name. the employee's full name is 'first_name last_name'.
    `BirthDate` datetime, -- Birth Date. the birth date of the employee.
    `Photo` blob, -- the photo of the employee.
    `Notes` text, -- some additional information of the employee.
    primary key (EmployeeID)
);

CREATE TABLE Categories (
    `CategoryID` integer, -- Category ID. the unique id for the category.
    `CategoryName` text, -- Category Name. the category name.
    `Description` text, -- the detailed description of the category.
    primary key (CategoryID)
);

CREATE TABLE Suppliers (
    `SupplierID` integer, -- Supplier ID. the unique id for suppliers.
    `SupplierName` text, -- Supplier Name. the supplier name.
    `ContactName` text, -- Contact Name. the contact person's name representing the company.
    `Address` text, -- the address of the supplier.
    `City` text, -- the city where the supplier is located.
    `PostalCode` text, -- Postal Code. the postal code.
    `Country` text, -- the country.
    `Phone` text, -- the phone number.
    primary key (SupplierID)
);

CREATE TABLE Customers (
    `CustomerID` text, -- Customer ID. the unique id for customers.
    `CustomerName` text, -- Customer Name. the customer name.
    `ContactName` text, -- Contact Name. the contact person's name representing the company.
    `Address` text, -- the address of the customer.
    `City` text, -- the city where the customer is located.
    `PostalCode` text, -- Postal Code. the postal code.
    `Country` text, -- the country.
    primary key (CustomerID)
);

CREATE TABLE Shippers (
    `ShipperID` integer, -- Shipper ID. the unique id for shippers.
    `ShipperName` text, -- Shipper Name. the shipped company name.
    `Phone` text, -- the phone of the company.
    primary key (ShipperID)
);

