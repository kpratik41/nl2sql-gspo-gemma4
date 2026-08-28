CREATE TABLE orderdetails (
    `orderNumber` integer, -- order number.
    `productCode` text, -- product code.
    `quantityOrdered` integer, -- quantity ordered.
    `priceEach` real, -- price for each. total price = quantityOrdered x priceEach.
    `orderLineNumber` integer, -- order Line Number.
    primary key (orderNumber, productCode)
);

CREATE TABLE offices (
    `officeCode` text, -- office code. unique ID of the office.
    `city` text, -- No description.
    `phone` text, -- phone number.
    `addressLine1` text, -- addressLine1.
    `addressLine2` text, -- addressLine2. addressLine1 + addressLine2 = entire address.
    `state` text, -- No description.
    `country` text, -- country.
    `postalCode` text, -- postalCode.
    `territory` text, -- territory.
    primary key (officeCode)
);

CREATE TABLE payments (
    `customerNumber` integer, -- customer number.
    `checkNumber` text, -- check Number.
    `paymentDate` date, -- payment Date.
    `amount` real, -- amount.
    primary key (customerNumber, checkNumber),
    foreign key (customerNumber) references customers(customerNumber)
);

CREATE TABLE products (
    `productCode` text, -- product code. unique product code.
    `productName` text, -- product name.
    `productLine` text, -- product line. product line name.
    `productScale` text, -- product scale.
    `productVendor` text, -- product vendor.
    `productDescription` text, -- product description.
    `quantityInStock` integer, -- quantity in stock.
    `buyPrice` real, -- buy price. buy price from vendors.
    `MSRP` real, -- Manufacturer Suggested Retail Price. expected profits: msrp - buyPrice.
    primary key (productCode),
    foreign key (productLine) references productlines(productLine)
);

CREATE TABLE employees (
    `employeeNumber` integer, -- Employee Number. unique string ID of the employees.
    `lastName` text, -- last name. last name of employees.
    `firstName` text, -- first name. first name of employees.
    `extension` text, -- extension number.
    `email` text, -- email.
    `officeCode` text, -- office code. office code of the employees.
    `reportsTo` integer, -- reports to. represents for organization structure such as who reports to whom. "reportsTO" is the leader of the "employeeNumber".
    `jobTitle` text, -- job title.
    primary key (employeeNumber),
    foreign key (officeCode) references offices(officeCode),
    foreign key (reportsTo) references employees(employeeNumber)
);

CREATE TABLE customers (
    `customerNumber` integer, -- customer number. unique id number of customer.
    `customerName` text, -- customer name. the name when the customer registered.
    `contactLastName` text, -- contact last name.
    `contactFirstName` text, -- contact first name.
    `phone` text, -- phone.
    `addressLine1` text, -- addressLine1.
    `addressLine2` text, -- addressLine2. addressLine1 + addressLine2 = entire address.
    `city` text, -- city.
    `state` text, -- state.
    `postalCode` text, -- postalCode.
    `country` text, -- country.
    `salesRepEmployeeNumber` integer, -- sales representative employee number.
    `creditLimit` real, -- credit limit.
    primary key (customerNumber),
    foreign key (salesRepEmployeeNumber) references employees(employeeNumber)
);

CREATE TABLE orders (
    `orderNumber` integer, -- order number. unique order number.
    `orderDate` date, -- order date.
    `requiredDate` date, -- required Date.
    `shippedDate` date, -- shipped date. shipped Date.
    `status` text, -- status.
    `comments` text, -- comments.
    `customerNumber` integer, -- customer number.
    primary key (orderNumber),
    foreign key (customerNumber) references customers(customerNumber)
);

CREATE TABLE productlines (
    `productLine` text, -- product line. unique product line name.
    `textDescription` text, -- text description.
    `htmlDescription` text, -- html description.
    `image` blob, -- image.
    primary key (productLine)
);

