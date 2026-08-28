CREATE TABLE categories (
    categoryid INTEGER NOT NULL, -- example: [1, 2, 3]
    categoryname TEXT NOT NULL, -- example: ['Beverages', 'Condiments', 'Confections']
    description TEXT, -- example: ['Soft drinks, coffees, teas, beers, and ales', 'Sweet and savory sauces, relishes, spreads, and seasonings', 'Desserts, candies, and sweet breads']
    picture BLOB -- example: ['\x']
)

CREATE TABLE customercustomerdemo (
    customerid TEXT NOT NULL, -- example: NULL
    customertypeid TEXT NOT NULL -- example: NULL
)

CREATE TABLE customerdemographics (
    customertypeid TEXT NOT NULL, -- example: NULL
    customerdesc TEXT -- example: NULL
)

CREATE TABLE customers (
    customerid TEXT NOT NULL, -- example: ['ALFKI', 'ANATR', 'ANTON']
    companyname TEXT NOT NULL, -- example: ['Alfreds Futterkiste', 'Ana Trujillo Emparedados y helados', 'Antonio Moreno Taquería']
    contactname TEXT, -- example: ['Maria Anders', 'Ana Trujillo', 'Antonio Moreno']
    contacttitle TEXT, -- example: ['Sales Representative', 'Owner', 'Order Administrator']
    address TEXT, -- example: ['Obere Str. 57', 'Avda. de la Constitución 2222', 'Mataderos  2312']
    city TEXT, -- example: ['Berlin', 'México D.F.', 'London']
    region TEXT, -- example: ['BC', 'SP', 'OR']
    postalcode TEXT, -- example: ['12209', '05021', '05023']
    country TEXT, -- example: ['Germany', 'Mexico', 'UK']
    phone TEXT, -- example: ['030-0074321', '(5) 555-4729', '(5) 555-3932']
    fax TEXT -- example: ['030-0076545', '(5) 555-3745', '(171) 555-6750']
)

CREATE TABLE employees (
    employeeid INTEGER NOT NULL, -- example: [1, 2, 3]
    lastname TEXT NOT NULL, -- example: ['Davolio', 'Fuller', 'Leverling']
    firstname TEXT NOT NULL, -- example: ['Nancy', 'Andrew', 'Janet']
    title TEXT, -- example: ['Sales Representative', 'Vice President, Sales', 'Sales Manager']
    titleofcourtesy TEXT, -- example: ['Ms.', 'Dr.', 'Mrs.']
    birthdate DATE, -- example: ['1948-12-08', '1952-02-19', '1963-08-30']
    hiredate DATE, -- example: ['1992-05-01', '1992-08-14', '1992-04-01']
    address TEXT, -- example: ['507 - 20th Ave. E.\nApt. 2A', '908 W. Capital Way', '722 Moss Bay Blvd.']
    city TEXT, -- example: ['Seattle', 'Tacoma', 'Kirkland']
    region TEXT, -- example: ['WA']
    postalcode TEXT, -- example: ['98122', '98401', '98033']
    country TEXT, -- example: ['USA', 'UK']
    homephone TEXT, -- example: ['(206) 555-9857', '(206) 555-9482', '(206) 555-3412']
    extension TEXT, -- example: ['5467', '3457', '3355']
    photo BLOB, -- example: ['\x']
    notes TEXT, -- example (truncated): ['Education includes a BA in psychology from Colorado State University in 1970.  She also completed Th...', 'Andrew received his BTS commercial in 1974 and a Ph.D. in international marketing from the Universit...', 'Janet has a BS degree in chemistry from Boston College (1984).  She has also completed a certificate...']
    reportsto INTEGER, -- example: [2, 5]
    photopath TEXT -- example: ['http://accweb/emmployees/davolio.bmp', 'http://accweb/emmployees/fuller.bmp', 'http://accweb/emmployees/leverling.bmp']
)

CREATE TABLE employeeterritories (
    employeeid INTEGER NOT NULL, -- example: [1, 2, 3]
    territoryid TEXT NOT NULL -- example: ['06897', '19713', '01581']
)

CREATE TABLE order_details (
    orderid INTEGER NOT NULL, -- example: [10248, 10249, 10250]
    productid INTEGER NOT NULL, -- example: [11, 42, 72]
    unitprice REAL NOT NULL, -- example: [14.0, 9.80000019, 34.7999992]
    quantity INTEGER NOT NULL, -- example: [12, 10, 5]
    discount REAL NOT NULL -- example: [0.0, 0.150000006, 0.0500000007]
)

CREATE TABLE orders (
    orderid INTEGER NOT NULL, -- example: [10248, 10249, 10250]
    customerid TEXT, -- example: ['VINET', 'TOMSP', 'HANAR']
    employeeid INTEGER, -- example: [5, 6, 4]
    orderdate DATE, -- example: ['1996-07-04', '1996-07-05', '1996-07-08']
    requireddate DATE, -- example: ['1996-08-01', '1996-08-16', '1996-08-05']
    shippeddate DATE, -- example: ['1996-07-16', '1996-07-10', '1996-07-12']
    shipvia INTEGER, -- example: [3, 1, 2]
    freight REAL, -- example: [32.3800011, 11.6099997, 65.8300018]
    shipname TEXT, -- example: ['Vins et alcools Chevalier', 'Toms Spezialitäten', 'Hanari Carnes']
    shipaddress TEXT, -- example: ['59 rue de l''Abbaye', 'Luisenstr. 48', 'Rua do Paço, 67']
    shipcity TEXT, -- example: ['Reims', 'Münster', 'Rio de Janeiro']
    shipregion TEXT, -- example: ['RJ', 'SP', 'Táchira']
    shippostalcode TEXT, -- example: ['51100', '44087', '05454-876']
    shipcountry TEXT -- example: ['France', 'Germany', 'Brazil']
)

CREATE TABLE products (
    productid INTEGER NOT NULL, -- example: [1, 2, 3]
    productname TEXT NOT NULL, -- example: ['Chai', 'Chang', 'Aniseed Syrup']
    supplierid INTEGER, -- example: [8, 1, 2]
    categoryid INTEGER, -- example: [1, 2, 7]
    quantityperunit TEXT, -- example: ['10 boxes x 30 bags', '24 - 12 oz bottles', '12 - 550 ml bottles']
    unitprice REAL, -- example: [18.0, 19.0, 10.0]
    unitsinstock INTEGER, -- example: [39, 17, 13]
    unitsonorder INTEGER, -- example: [0, 40, 70]
    reorderlevel INTEGER, -- example: [10, 25, 0]
    discontinued INTEGER NOT NULL -- example: [1, 0]
)

CREATE TABLE region (
    regionid INTEGER NOT NULL, -- example: [1, 2, 3]
    regiondescription TEXT NOT NULL -- example: ['Eastern', 'Western', 'Northern']
)

CREATE TABLE shippers (
    shipperid INTEGER NOT NULL, -- example: [1, 2, 3]
    companyname TEXT NOT NULL, -- example: ['Speedy Express', 'United Package', 'Federal Shipping']
    phone TEXT -- example: ['(503) 555-9831', '(503) 555-3199', '(503) 555-9931']
)

CREATE TABLE suppliers (
    supplierid INTEGER NOT NULL, -- example: [1, 2, 3]
    companyname TEXT NOT NULL, -- example: ['Exotic Liquids', 'New Orleans Cajun Delights', 'Grandma Kelly''s Homestead']
    contactname TEXT, -- example: ['Charlotte Cooper', 'Shelley Burke', 'Regina Murphy']
    contacttitle TEXT, -- example: ['Purchasing Manager', 'Order Administrator', 'Sales Representative']
    address TEXT, -- example: ['49 Gilbert St.', 'P.O. Box 78934', '707 Oxford Rd.']
    city TEXT, -- example: ['London', 'New Orleans', 'Ann Arbor']
    region TEXT, -- example: ['LA', 'MI', 'Asturias']
    postalcode TEXT, -- example: ['EC1 4SD', '70117', '48104']
    country TEXT, -- example: ['UK', 'USA', 'Japan']
    phone TEXT, -- example: ['(171) 555-2222', '(100) 555-4822', '(313) 555-5735']
    fax TEXT, -- example: ['(313) 555-3349', '(03) 444-6588', '031-987 65 91']
    homepage TEXT -- example: ['#CAJUN.HTM#', 'Mayumi''s (on the World Wide Web)#http://www.microsoft.com/accessdev/sampleapps/mayumi.htm#', 'Plutzer (on the World Wide Web)#http://www.microsoft.com/accessdev/sampleapps/plutzer.htm#']
)

CREATE TABLE territories (
    territoryid TEXT NOT NULL, -- example: ['01581', '01730', '01833']
    territorydescription TEXT NOT NULL, -- example: ['Westboro', 'Bedford', 'Georgetow']
    regionid INTEGER NOT NULL -- example: [1, 3, 4]
)

CREATE TABLE usstates (
    stateid INTEGER NOT NULL, -- example: [1, 2, 3]
    statename TEXT, -- example: ['Alabama', 'Alaska', 'Arizona']
    stateabbr TEXT, -- example: ['AL', 'AK', 'AZ']
    stateregion TEXT -- example: ['south', 'north', 'west']
)

CREATE TABLE customergroupthreshold (
    groupname TEXT NOT NULL, -- example: ['Low', 'Medium', 'High']
    rangebottom DECIMAL NOT NULL, -- example: [0, 1000, 5000]
    rangetop DECIMAL NOT NULL -- example: [999.9999, 4999.9999, 9999.9999]
)

