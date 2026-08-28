CREATE TABLE product (
    `Product ID` text, -- the id of products.
    `Product Name` text, -- the name of products.
    `Category` text, -- the categories of products.  Furniture.  Office Supplies.  Technology.
    `Sub-Category` text, -- the sub-categories of products.  Bookcases.  Chairs.  Furnishings.  Tables.  Appliances.  Art.  Binders.  Envelopes.  Fasteners.  Labels.  Paper.  Storage.  Supplies.  Accessories.  Copiers.  Machines.  Phones.
    `Region` text, -- the region where products are sold.  Central:.  East:.  West:.  South:.
    primary key ("Product ID", Region)
);

CREATE TABLE west_superstore (
    `Row ID` integer, -- row id. the unique id for rows.
    `Order ID` text, -- order id. the unique identifier for the order.
    `Order Date` date, -- order date. the date of the order. yyyy-mm-dd.
    `Ship Date` date, -- ship date. the date of the shipment. yyyy-mm-dd. 'shipment time' refers to the time interval between order_date and ship_date.
    `Ship Mode` text, -- ship mode. the ship mode of the order. First Class / Second Class / Standard Class. Among three ship modes, First Class has the fastest delivery speed, followed by Second Class and the speed of the Standard Class is the slowest.
    `Customer ID` text, -- customer id. the id of the customer.
    `Region` text, -- region of the customer's address.
    `Product ID` text, -- product id. the id of the product.
    `Sales` real, -- the sales of the product.
    `Quantity` integer, -- the quantity of the product.
    `Discount` real, -- the discount of the product. original price = sales / (1- discount).
    `Profit` real, -- the profit that the company got by selling the product. total cost of products = sales / (1- discount) * quantity - profit. deficiency: if the value is negative.
    primary key ("Row),
    foreign key ("Customer ID", Region) references people("Customer ID",Region),
    foreign key ("Product ID", Region) references product("Product ID",Region)
);

CREATE TABLE central_superstore (
    `Row ID` integer, -- row id. the unique id for rows.
    `Order ID` text, -- order id. the unique identifier for the order.
    `Order Date` date, -- order date. the date of the order. yyyy-mm-dd.
    `Ship Date` date, -- ship date. the date of the shipment. yyyy-mm-dd. 'shipment time' refers to the time interval between order_date and ship_date.
    `Ship Mode` text, -- ship mode. the ship mode of the order. First Class / Second Class / Standard Class. Among three ship modes, First Class has the fastest delivery speed, followed by Second Class and the speed of the Standard Class is the slowest.
    `Customer ID` text, -- customer id. the id of the customer.
    `Region` text, -- region of the customer's address.
    `Product ID` text, -- product id. the id of the product.
    `Sales` real, -- the sales of the product.
    `Quantity` integer, -- the quantity of the product.
    `Discount` real, -- the discount of the product. original price = sales / (1- discount).
    `Profit` real, -- the profit that the company got by selling the product. total cost of products = sales / (1- discount) * quantity - profit. deficiency: if the value is negative.
    primary key ("Row),
    foreign key ("Customer ID", Region) references people("Customer ID",Region),
    foreign key ("Product ID", Region) references product("Product ID",Region)
);

CREATE TABLE south_superstore (
    `Row ID` integer, -- row id. the unique id for rows.
    `Order ID` text, -- order id. the unique identifier for the order.
    `Order Date` date, -- order date. the date of the order. yyyy-mm-dd.
    `Ship Date` date, -- ship date. the date of the shipment. yyyy-mm-dd. 'shipment time' refers to the time interval between order_date and ship_date.
    `Ship Mode` text, -- ship mode. the ship mode of the order. First Class / Second Class / Standard Class. Among three ship modes, First Class has the fastest delivery speed, followed by Second Class and the speed of the Standard Class is the slowest.
    `Customer ID` text, -- customer id. the id of the customer.
    `Region` text, -- region of the customer's address.
    `Product ID` text, -- product id. the id of the product.
    `Sales` real, -- the sales of the product.
    `Quantity` integer, -- the quantity of the product.
    `Discount` real, -- the discount of the product. original price = sales / (1- discount).
    `Profit` real, -- the profit that the company got by selling the product. total cost of products = sales / (1- discount) * quantity - profit. deficiency: if the value is negative.
    primary key ("Row),
    foreign key ("Customer ID", Region) references people("Customer ID",Region),
    foreign key ("Product ID", Region) references product("Product ID",Region)
);

CREATE TABLE east_superstore (
    `Row ID` integer, -- row id. the unique id for rows.
    `Order ID` text, -- order id. the unique identifier for the order.
    `Order Date` date, -- order date. the date of the order. yyyy-mm-dd.
    `Ship Date` date, -- ship date. the date of the shipment. yyyy-mm-dd. 'shipment time' refers to the time interval between order_date and ship_date.
    `Ship Mode` text, -- ship mode. the ship mode of the order. First Class / Second Class / Standard Class. Among three ship modes, First Class has the fastest delivery speed, followed by Second Class and the speed of the Standard Class is the slowest.
    `Customer ID` text, -- customer id. the id of the customer.
    `Region` text, -- region of the customer's address.
    `Product ID` text, -- product id. the id of the product.
    `Sales` real, -- the sales of the product.
    `Quantity` integer, -- the quantity of the product.
    `Discount` real, -- the discount of the product. original price = sales / (1- discount).
    `Profit` real, -- the profit that the company got by selling the product. total cost of products = sales / (1- discount) * quantity - profit. deficiency: if the value is negative.
    primary key ("Row),
    foreign key ("Customer ID", Region) references people("Customer ID",Region),
    foreign key ("Product ID", Region) references product("Product ID",Region)
);

CREATE TABLE people (
    `Customer ID` text, -- the id of the customers.
    `Customer Name` text, -- the name of the customers.
    `Segment` text, -- the segment that the customers belong to.  consumer.  home office: synonym: headquarter.  corporate.
    `Country` text, -- the country of people.
    `City` text, -- the city of people.
    `State` text, -- the state of people. please mention its full name in the question, by referring to. https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971. e.g., New York --> NY.
    `Postal Code` integer, -- the postal code.
    `Region` text, -- the region of people.  Central:.  East:.  West:.  South:.
    primary key ("Customer ID", Region)
);

