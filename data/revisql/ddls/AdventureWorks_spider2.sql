CREATE TABLE salesperson (
 businessentityid INTEGER, -- example: [274, 275, 276]
 territoryid INTEGER, -- example: [2, 4, 3]
 salesquota INTEGER, -- example: [300000, 250000]
 bonus INTEGER, -- example: [0, 4100, 2000]
 commissionpct FLOAT, -- example: [0.0, 0.012, 0.015]
 salesytd FLOAT, -- example: [559697.5639, 3763178.1787, 4251368.5497]
 saleslastyear FLOAT, -- example: [0.0, 1750406.4785, 1439156.0291]
 rowguid TEXT, -- example: ['48754992-9ee0-4c0e-8c94-9451604e3e02', '1e0a7274-3064-4f58-88ee-4c6586c87169', '4dd9eee4-8e81-4f8c-af97-683394c1f7c0']
 modifieddate DATE -- example: ['2010-12-28 00:00:00', '2011-05-24 00:00:00', '2012-09-23 00:00:00']
 )

CREATE TABLE product (
    productid INTEGER, -- example: [1, 2, 3]
    NAME TEXT, -- example: ['Adjustable Race', 'Bearing Ball', 'BB Ball Bearing']
    productnumber TEXT, -- example: ['AR-5381', 'BA-8327', 'BE-2349']
    makeflag BOOLEAN, -- example: ['f', 't']
    finishedgoodsflag BOOLEAN, -- example: ['f', 't']
    color TEXT, -- example: ['Black', 'Silver', 'Red']
    safetystocklevel INTEGER, -- example: [1000, 800, 500]
    reorderpoint INTEGER, -- example: [750, 600, 375]
    standardcost FLOAT, -- example: [0.0, 98.77, 108.99]
    listprice FLOAT, -- example: [0.0, 133.34, 147.14]
    size TEXT, -- example: ['58', 'M', 'L']
    sizeunitmeasurecode TEXT, -- example: ['CM ']
    weightunitmeasurecode TEXT, -- example: ['G  ', 'LB ']
    weight FLOAT, -- example: [435.0, 450.0, 400.0]
    daystomanufacture INTEGER, -- example: [0, 1, 2]
    productline TEXT, -- example: ['R ', 'S ', 'M ']
    class TEXT, -- example: ['L ', 'M ', 'H ']
    style TEXT, -- example: ['U ', 'W ', 'M ']
    productsubcategoryid INTEGER, -- example: [14, 31, 23]
    productmodelid INTEGER, -- example: [6, 33, 18]
    sellstartdate DATE, -- example: ['2008-04-30 00:00:00', '2011-05-31 00:00:00', '2012-05-30 00:00:00']
    sellenddate DATE, -- example: ['2012-05-29 00:00:00', '2013-05-29 00:00:00']
    discontinueddate DATE, -- example: NULL
    rowguid TEXT, -- example: ['694215b7-08f7-4c0d-acb1-d734ba44c0c8', '58ae3c20-4f3a-4749-a7d4-d568806cc537', '9c21aed2-5bfa-4f18-bcb8-f11638dc2e4e']
    modifieddate DATE -- example: ['2014-02-08 10:01:36.827', '2014-02-08 10:03:55.51']
    )

CREATE TABLE productmodelproductdescriptionculture (
    productmodelid INTEGER, -- example: [1, 2, 3]
    productdescriptionid INTEGER, -- example: [1199, 1467, 1589]
    cultureid TEXT, -- example: ['en', 'ar', 'fr']
    modifieddate DATE -- example: ['2013-04-30 00:00:00']
    )

CREATE TABLE productdescription (
    productdescriptionid INTEGER, -- example: [3, 4, 5]
    description TEXT, -- example: ['Chromoly steel.', 'Aluminum alloy cups; large diameter spindle.', 'Aluminum alloy cups and a hollow axle.']
    rowguid TEXT, -- example: ['301eed3a-1a82-4855-99cb-2afe8290d641', 'dfeba528-da11-4650-9d86-cafda7294eb0', 'f7178da7-1a7e-4997-8470-06737181305e']
    modifieddate DATE -- example: ['2013-04-30 00:00:00', '2014-02-08 10:32:17.973']
    )

CREATE TABLE productreview (
    productreviewid INTEGER, -- example: [1, 2, 3]
    productid INTEGER, -- example: [709, 937, 798]
    reviewername TEXT, -- example: ['John Smith', 'David', 'Jill']
    reviewdate DATE, -- example: ['2013-09-18 00:00:00', '2013-11-13 00:00:00', '2013-11-15 00:00:00']
    emailaddress TEXT, -- example: ['john@fourthcoffee.com', 'david@graphicdesigninstitute.com', 'jill@margiestravel.com']
    rating INTEGER, -- example: [5, 4, 2]
    comments TEXT, -- example (truncated): ['I can''t believe I''m singing the praises of a pair of socks, but I just came back from a grueling\n3-...', 'A little on the heavy side, but overall the entry/exit is easy in all conditions. I''ve used these pe...', 'Maybe it''s just because I''m new to mountain biking, but I had a terrible time getting use\nto these ...']
    modifeddate DATE -- example: NULL
    , modifieddate VARCHAR(19)) -- example: ['2013-09-18 00:00:00', '2013-11-13 00:00:00', '2013-11-15 00:00:00']

CREATE TABLE productcategory (
    productcategoryid INTEGER, -- example: [1, 2, 3]
    name TEXT, -- example: ['Bikes', 'Components', 'Clothing']
    rowguid TEXT, -- example: ['cfbda25c-df71-47a7-b81b-64ee161aa37c', 'c657828d-d808-4aba-91a3-af2ce02300e9', '10a7c342-ca82-48d4-8a38-46a2eb089b74']
    modifieddate DATE -- example: ['2008-04-30 00:00:00']
    )

CREATE TABLE productsubcategory (
    productsubcategoryid INTEGER, -- example: [1, 2, 3]
    productcategoryid INTEGER, -- example: [1, 2, 3]
    name TEXT, -- example: ['Mountain Bikes', 'Road Bikes', 'Touring Bikes']
    rowguid TEXT, -- example: ['2d364ade-264a-433c-b092-4fcbf3804e01', '000310c0-bcc8-42c4-b0c3-45ae611af06b', '02c5061d-ecdc-4274-b5f1-e91d76bc3f37']
    modifieddate DATE -- example: ['2008-04-30 00:00:00']
    )

CREATE TABLE salesorderdetail (
    salesorderid INTEGER, -- example: [43659, 43660, 43661]
    salesorderdetailid INTEGER, -- example: [1, 2, 3]
    carriertrackingnumber TEXT, -- example: ['4911-403C-98', '6431-4D57-83', '4E0A-4F89-AE']
    orderqty INTEGER, -- example: [1, 3, 2]
    productid INTEGER, -- example: [776, 777, 778]
    specialofferid INTEGER, -- example: [1, 2, 3]
    unitprice FLOAT, -- example: [2024.994, 2039.994, 28.8404]
    unitpricediscount FLOAT, -- example: [0.0, 0.02, 0.05]
    rowguid TEXT, -- example: ['b207c96d-d9e6-402b-8470-2cc176c42283', '7abb600d-1e77-41be-9fe5-b9142cfc08fa', '475cf8c6-49f6-486e-b0ad-afc6a50cdd2f']
    modifieddate DATE -- example: ['2011-05-31 00:00:00', '2011-06-01 00:00:00', '2011-06-02 00:00:00']
    )

CREATE TABLE salesorderheader (
    salesorderid INTEGER, -- example: [43659, 43660, 43661]
    revisionnumber INTEGER, -- example: [8, 9]
    orderdate DATE, -- example: ['2011-05-31 00:00:00', '2011-06-01 00:00:00', '2011-06-02 00:00:00']
    duedate DATE, -- example: ['2011-06-12 00:00:00', '2011-06-13 00:00:00', '2011-06-14 00:00:00']
    shipdate DATE, -- example: ['2011-06-07 00:00:00', '2011-06-08 00:00:00', '2011-06-09 00:00:00']
    STATUS TEXT, -- example: ['5']
    onlineorderflag BOOLEAN, -- example: ['f', 't']
    purchaseordernumber TEXT, -- example: ['PO522145787', 'PO18850127500', 'PO18473189620']
    accountnumber TEXT, -- example: ['10-4020-000676', '10-4020-000117', '10-4020-000442']
    customerid INTEGER, -- example: [29825, 29672, 29734]
    salespersonid INTEGER, -- example: [279, 282, 276]
    territoryid INTEGER, -- example: [5, 6, 4]
    billtoaddressid INTEGER, -- example: [985, 921, 517]
    shiptoaddressid INTEGER, -- example: [985, 921, 517]
    shipmethodid INTEGER, -- example: [5, 1]
    creditcardid INTEGER, -- example: [16281, 5618, 1346]
    creditcardapprovalcode TEXT, -- example: ['105041Vi84182', '115213Vi29411', '85274Vi6854']
    currencyrateid INTEGER, -- example: [4, 8, 2]
    subtotal FLOAT, -- example: [20565.6206, 1294.2529, 32726.4786]
    taxamt FLOAT, -- example: [1971.5149, 124.2483, 3153.7696]
    freight FLOAT, -- example: [616.0984, 38.8276, 985.553]
    totaldue FLOAT, -- example: [23153.2339, 1457.3288, 36865.8012]
    comment TEXT, -- example: NULL
    rowguid TEXT, -- example: ['79b65321-39ca-4115-9cba-8fe0903e12e6', '738dc42d-d03b-48a1-9822-f95a67ea7389', 'd91b9131-18a4-4a11-bc3a-90b6f53e9d74']
    modifieddate DATE -- example: ['2011-06-07 00:00:00', '2011-06-08 00:00:00', '2011-06-09 00:00:00']
    )

CREATE TABLE salesterritory (
    territoryid INTEGER, -- example: [1, 2, 3]
    name TEXT, -- example: ['Northwest', 'Northeast', 'Central']
    countryregioncode TEXT, -- example: ['US', 'CA', 'FR']
    "group" TEXT, -- example: ['North America', 'Europe', 'Pacific']
    salesytd FLOAT, -- example: [7887186.7882, 2402176.8476, 3072175.118]
    saleslastyear FLOAT, -- example: [3298694.4938, 3607148.9371, 3205014.0767]
    costytd FLOAT, -- example: [0.0]
    costlastyear FLOAT, -- example: [0.0]
    rowguid TEXT, -- example: ['43689a10-e30b-497f-b0de-11de20267ff7', '00fb7309-96cc-49e2-8363-0a1ba72486f2', 'df6e7fd8-1a8d-468c-b103-ed8addb452c1']
    modifieddate DATE -- example: ['2008-04-30 00:00:00']
    )

CREATE TABLE countryregioncurrency (
    countryregioncode TEXT, -- example: ['AE', 'AR', 'AT']
    currencycode TEXT, -- example: ['AED', 'ARS', 'ATS']
    modifieddate DATE -- example: ['2014-02-08 10:17:21.51', '2008-04-30 00:00:00']
    )

CREATE TABLE currencyrate (
    currencyrateid INTEGER, -- example: [1, 2, 3]
    currencyratedate DATE, -- example: ['2011-05-31 00:00:00', '2011-06-01 00:00:00', '2011-06-02 00:00:00']
    fromcurrencycode TEXT, -- example: ['USD']
    tocurrencycode TEXT, -- example: ['ARS', 'AUD', 'BRL']
    averagerate FLOAT, -- example: [1.0, 1.5491, 1.9379]
    endofdayrate FLOAT, -- example: [1.0002, 1.55, 1.9419]
    modifieddate DATE -- example: ['2011-05-31 00:00:00', '2011-06-01 00:00:00', '2011-06-02 00:00:00']
    )

CREATE TABLE SalesPersonQuotaHistory (
    BusinessEntityID INTEGER, -- example: [274, 275, 276]
    QuotaDate TEXT, -- example: ['2011-05-31 00:00:00', '2011-08-31 00:00:00', '2011-12-01 00:00:00']
    SalesQuota REAL, -- example: [28000.0, 7000.0, 91000.0]
    rowguid TEXT, -- example: ['{99109BBF-8693-4587-BC23-6036EC89E1BE}', '{DFD01444-8900-461C-8D6F-04598DAE01D4}', '{0A69F453-9689-4CCF-A08C-C644670F5668}']
    ModifiedDate TEXT -- example: ['2011-04-16 00:00:00', '2011-07-17 00:00:00', '2011-10-17 00:00:00']
)

