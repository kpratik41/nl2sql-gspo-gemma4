CREATE TABLE PurchaseOrderHeader (
    `PurchaseOrderID` integer, -- Purchase Order ID. The unique id number identifying purchase order.
    `RevisionNumber` integer, -- Revision Number. Incremental number to track changes to the purchase order over time.
    `Status` integer, -- Order current status. 1 = Pending; 2 = Approved; 3 = Rejected; 4 = Complete Default: 1.
    `EmployeeID` integer, -- Employee ID. Employee who created the purchase order.
    `VendorID` integer, -- Vendor ID. Vendor with whom the purchase order is placed.
    `ShipMethodID` integer, -- Ship Method ID. Shipping method.
    `OrderDate` datetime, -- Order Date. Purchase order creation date. Default: getdate().
    `ShipDate` datetime, -- Ship Date. Estimated shipment date from the vendor.
    `SubTotal` real, -- Purchase order subtotal. Computed as SUM (PurchaseOrderDetail.LineTotal)for the appropriate PurchaseOrderID.
    `TaxAmt` real, -- Tax Amount. Tax amount.
    `Freight` real, -- Shipping cost.
    `TotalDue` real, -- Total Due. Total due to vendor. Computed as Subtotal + TaxAmt + Freight. Computed: isnull(([SubTotal]+[TaxAmt])+[Freight],(0)).
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (PurchaseOrderID),
    foreign key (EmployeeID) references Employee(BusinessEntityID),
    foreign key (VendorID) references Vendor(BusinessEntityID),
    foreign key (ShipMethodID) references ShipMethod(ShipMethodID)
);

CREATE TABLE CountryRegion (
    `CountryRegionCode` text, -- Country Region Code. The unique id number identifying Country Region ISO standard code for countries and regions.
    `Name` text, -- Country or region name.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CountryRegionCode)
);

CREATE TABLE EmailAddress (
    `BusinessEntityID` integer, -- Business Entity ID. The id number identifying the person associated with this email address.
    `EmailAddressID` integer, -- Email Address ID. The ID of this email address.
    `EmailAddress` text, -- Email Address. The E-mail address for the person.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (EmailAddressID, BusinessEntityID),
    foreign key (BusinessEntityID) references Person(BusinessEntityID)
);

CREATE TABLE BillOfMaterials (
    `BillOfMaterialsID` integer, -- bill of materials id. The unique number identifying bill of materials. Primary key for BillOfMaterials records. Identity / Auto increment column.
    `ProductAssemblyID` integer, -- product assembly id. Parent product identification number.
    `ComponentID` integer, -- component ID. Component identification number.
    `StartDate` datetime, -- start date. Date the component started being used in the assembly item.
    `EndDate` datetime, -- end date. Date the component stopped being used in the assembly item. 1. assembly item duration = (EndDate - StartDate). 2. if EndDate is null, it means the assembly item doesn't finish (still going on).
    `UnitMeasureCode` text, -- unit measure code. Standard code identifying the unit of measure for the quantity.
    `BOMLevel` integer, -- bill of materials level. Indicates the depth the component is from its parent (column2)(AssemblyID).
    `PerAssemblyQty` real, -- per assembly quantity. Quantity of the component needed to create the assembly.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BillOfMaterialsID)
);

CREATE TABLE BusinessEntity (
    `BusinessEntityID` integer, -- business entity id. Unique number of identifying business entity. Primary key for all customers, vendors, and employees. Identity / Auto increment column.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID)
);

CREATE TABLE EmployeeDepartmentHistory (
    `BusinessEntityID` integer, -- Business Entity ID. Employee identification number.
    `DepartmentID` integer, -- DepartmentI D. Department in which the employee worked including currently.
    `ShiftID` integer, -- Shift ID. Identifies which 8-hour shift the employee works.
    `StartDate` date, -- Start Date. Date the employee started working in the department.
    `EndDate` date, -- End Date. Date the employee ended working in the department.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, StartDate, DepartmentID, ShiftID)
);

CREATE TABLE PhoneNumberType (
    `PhoneNumberTypeID` integer, -- Phone Number Type ID. The id number identifying the telephone number type records.
    `Name` text, -- Phone Number. Name of the telephone number type.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (PhoneNumberTypeID)
);

CREATE TABLE SalesOrderDetail (
    `SalesOrderID` integer, -- No description.
    `SalesOrderDetailID` integer, -- No description.
    `CarrierTrackingNumber` text, -- No description.
    `OrderQty` integer, -- No description.
    `ProductID` integer, -- No description.
    `SpecialOfferID` integer, -- No description.
    `UnitPrice` real, -- No description.
    `UnitPriceDiscount` real, -- No description.
    `LineTotal` real, -- No description.
    `rowguid` text, -- No description.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated.
    primary key (SalesOrderDetailID),
    foreign key (SalesOrderID) references SalesOrderHeader(SalesOrderID),
    foreign key (SpecialOfferID, ProductID) references SpecialOfferProduct(SpecialOfferID, ProductID)
);

CREATE TABLE SalesTaxRate (
    `SalesTaxRateID` integer, -- Sales Tax Rate ID. Unique id number identifying sales tax records.
    `StateProvinceID` integer, -- State Province ID. id number identifying state province.
    `TaxType` integer, -- Tax Type. 1 = Tax applied to retail transactions, 2 = Tax applied to wholesale transactions, 3 = Tax applied to all sales (retail and wholesale) transactions.
    `TaxRate` real, -- Tax Rate. Tax rate amount.
    `Name` text, -- Tax rate description. if there's "+" in the value, it means this sales are charged by multiple types of tax.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (SalesTaxRateID),
    foreign key (StateProvinceID) references StateProvince(StateProvinceID)
);

CREATE TABLE Shift (
    `ShiftID` integer, -- The unique number identifying the shift.
    `Name` text, -- Shift description.
    `StartTime` text, -- Start Time. Shift start time.
    `EndTime` text, -- End Time. Shift end time.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ShiftID)
);

CREATE TABLE ProductDocument (
    `ProductID` integer, -- Product ID. The id number identifying the product.
    `DocumentNode` text, -- Document Node. The hierarchy id number identifying the Document node. Document identification number.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductID, DocumentNode),
    foreign key (ProductID) references Product(ProductID),
    foreign key (DocumentNode) references Document(DocumentNode)
);

CREATE TABLE BusinessEntityContact (
    `BusinessEntityID` integer, -- Business Entity ID. The id number identifying the Business Entity ID.
    `PersonID` integer, -- Person ID. The id number identifying the Person ID.
    `ContactTypeID` integer, -- Contact Type ID. The id number identifying the contact type ID.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, PersonID, ContactTypeID),
    foreign key (BusinessEntityID) references BusinessEntity(BusinessEntityID),
    foreign key (ContactTypeID) references ContactType(ContactTypeID),
    foreign key (PersonID) references Person(BusinessEntityID)
);

CREATE TABLE Department (
    `DepartmentID` integer, -- Department ID. The unique id number identifying the department.
    `Name` text, -- Name of the department.
    `GroupName` text, -- Group Name. Name of the group to which the department belongs.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (DepartmentID)
);

CREATE TABLE Location (
    `LocationID` integer, -- Location ID. The unique id number identifying the job candidates.
    `Name` text, -- Location description.
    `CostRate` real, -- Cost Rate. Standard hourly cost of the manufacturing location. Default: 0.00.
    `Availability` real, -- Work capacity (in hours) of the manufacturing location. Default: 0.00.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (LocationID)
);

CREATE TABLE WorkOrder (
    `WorkOrderID` integer, -- Work Order ID. The unique id number identifying work order.
    `ProductID` integer, -- Product ID. Product identification number.
    `OrderQty` integer, -- Order Quantity. Product quantity to build.
    `StockedQty` integer, -- Stocked Quantity. Quantity built and put in inventory. Computed: isnull([OrderQty]-[ScrappedQty],(0)).
    `ScrappedQty` integer, -- Scrapped Quantity. Quantity that failed inspection.
    `StartDate` datetime, -- Start Date. Work order start date.
    `EndDate` datetime, -- End Date. Work order end date.
    `DueDate` datetime, -- Due Date. Work order due date.
    `ScrapReasonID` integer, -- Scrap Reason ID. Reason for inspection failure.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (WorkOrderID),
    foreign key (ProductID) references Product(ProductID),
    foreign key (ScrapReasonID) references ScrapReason(ScrapReasonID)
);

CREATE TABLE Product (
    `ProductID` integer, -- Product ID. The unique id number identifying the product.
    `Name` text, -- Name of the product.
    `ProductNumber` text, -- Product Number. The unique product identification number.
    `MakeFlag` integer, -- Make Flag. The source of product make. 0 = Product is purchased, 1 = Product is manufactured in-house. Default: 1.
    `FinishedGoodsFlag` integer, -- Finished Goods Flag. Whether the product is salable or not. 0 = Product is not a salable item. 1 = Product is salable. Default: 1.
    `Color` text, -- Color. Product color.
    `SafetyStockLevel` integer, -- Safety Stock Level. The minimum inventory quantity.
    `ReorderPoint` integer, -- Reorder Point. Inventory level that triggers a purchase order or work order.
    `StandardCost` real, -- Standard Cost. Standard cost of the product.
    `ListPrice` real, -- List Price. Selling price. profit = ListPrice - StandardCost.
    `Size` text, -- Product size.
    `SizeUnitMeasureCode` text, -- Size Unit Measure Code. Unit of measure for Size column.
    `WeightUnitMeasureCode` text, -- Weight Unit Measure Code. Unit of measure for Weight column.
    `Weight` real, -- Product weight.
    `DaysToManufacture` integer, -- Days To Manufacture. Number of days required to manufacture the product.
    `ProductLine` text, -- Product Line. Product Routine. R = Road, M = Mountain, T = Touring, S = Standard.
    `Class` text, -- Product quality class. H = High, M = Medium, L = Low.
    `Style` text, -- Style. W = Womens, M = Mens, U = Universal.
    `ProductSubcategoryID` integer, -- Product Subcategory ID. Product is a member of this product subcategory.
    `ProductModelID` integer, -- Product Model ID. Product is a member of this product model.
    `SellStartDate` datetime, -- Sell Start Date. Date the product was available for sale.
    `SellEndDate` datetime, -- Sell End Date. Date the product was no longer available for sale.
    `DiscontinuedDate` datetime, -- Discontinued Date. Date the product was discontinued.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductID)
);

CREATE TABLE SalesReason (
    `SalesReasonID` integer, -- The unique number identifying SalesReason records.
    `Name` text, -- Sales reason description.
    `ReasonType` text, -- Category the sales reason belongs to.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (SalesReasonID)
);

CREATE TABLE ProductCategory (
    `ProductCategoryID` integer, -- Product Category ID. The unique id number identifying the product category.
    `Name` text, -- Category description.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductCategoryID)
);

CREATE TABLE SalesPersonQuotaHistory (
    `BusinessEntityID` integer, -- Business Entity ID. Sales person identification number.
    `QuotaDate` datetime, -- Quota Date. Sales quota date.
    `SalesQuota` real, -- Sales Quota. Sales quota amount.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, QuotaDate),
    foreign key (BusinessEntityID) references SalesPerson(BusinessEntityID)
);

CREATE TABLE ProductProductPhoto (
    `ProductID` integer, -- Product ID. Product identification number.
    `ProductPhotoID` integer, -- Product Photo ID. Product photo identification number.
    `Primary` integer, -- Whether this photo is the principal image or not. 0 = Photo is not the principal image. 1 = Photo is the principal image. staff can mention "principal" in the question in order to make the question more realistic.
    `ModifiedDate` datetime, -- Modified Date.
    primary key (ProductID, ProductPhotoID),
    foreign key (ProductID) references Product(ProductID),
    foreign key (ProductPhotoID) references ProductPhoto(ProductPhotoID)
);

CREATE TABLE Employee (
    `BusinessEntityID` integer, -- Business Entity ID. The id number identifying the employee.
    `NationalIDNumber` text, -- National ID Number. The national identification number such as a social security number.
    `LoginID` text, -- Login ID. Network login.
    `OrganizationNode` text, -- Organization Node. Where the employee is located in corporate hierarchy. Default: newid().
    `OrganizationLevel` integer, -- Organization Level. The depth of the employee in the corporate hierarchy. Computed: [OrganizationNode].[GetLevel]().
    `JobTitle` text, -- Job Title. Work title such as Buyer or Sales Representative.
    `BirthDate` date, -- Birth Date. Date of birth.
    `MaritalStatus` text, -- Marital Status. Whether this employee is married or not. M = Married, S = Single.
    `Gender` text, -- The gender of this employee. M = Male, F = Female.
    `HireDate` date, -- Hire Date. The employee was hired on this date.
    `SalariedFlag` integer, -- Salaried Flag. Job classification. 0 = Hourly, not exempt from collective bargaining. 1 = Salaried, exempt from collective bargaining.
    `VacationHours` integer, -- Vacation Hours. The number of available vacation hours.
    `SickLeaveHours` integer, -- Sick Leave Hours. The number of available sick leave hours.
    `CurrentFlag` integer, -- Current Flag. Whether this employee is active or not. 0 = Inactive, 1 = Active.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID),
    foreign key (BusinessEntityID) references Person(BusinessEntityID)
);

CREATE TABLE BusinessEntityAddress (
    `BusinessEntityID` integer , -- Business Entity ID. Number identifying the business entity.
    `AddressID` integer, -- Address ID. Number identifying the address.
    `AddressTypeID` integer, -- Address Type ID. Number identifying the address type.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, AddressID, AddressTypeID),
    foreign key (AddressID) references Address(AddressID),
    foreign key (AddressTypeID) references AddressType(AddressTypeID),
    foreign key (BusinessEntityID) references BusinessEntity(BusinessEntityID)
);

CREATE TABLE CountryRegionCurrency (
    `CountryRegionCode` text, -- Country Region Code. The id number identifying Country Region ISO standard code for countries and regions.
    `CurrencyCode` text, -- Currency Code. ISO standard currency code.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CountryRegionCode, CurrencyCode),
    foreign key (CountryRegionCode) references CountryRegion(CountryRegionCode),
    foreign key (CurrencyCode) references Currency(CurrencyCode)
);

CREATE TABLE UnitMeasure (
    `UnitMeasureCode` text, -- Unit Measure Code. The unique identifying numbers for measure.
    `Name` text, -- Unit of measure description.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (UnitMeasureCode)
);

CREATE TABLE ProductCostHistory (
    `ProductID` integer, -- Product ID. The id number identifying the product.
    `StartDate` date, -- Start Date. Product cost start date.
    `EndDate` date, -- End Date. Product cost end date.
    `StandardCost` real, -- Standard Cost. Standard cost of the product.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductID, StartDate),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE EmployeePayHistory (
    `BusinessEntityID` integer, -- Business Entity ID. Employee identification number.
    `RateChangeDate` datetime, -- Rate Change Date. Date the change in pay is effective.
    `Rate` real, -- Salary hourly rate.
    `PayFrequency` integer, -- Pay Frequency. Pay Frequency. 1 = Salary received monthly, 2 = Salary received biweekly.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, RateChangeDate)
);

CREATE TABLE Document (
    `DocumentNode` text, -- Document Node. The unique hierarchy id identifying the document nodes.
    `DocumentLevel` integer, -- Document Level. Depth in the document hierarchy. Computed: [DocumentNode].[GetLevel]().
    `Title` text, -- Title of the document.
    `Owner` integer, -- Employee who controls the document.
    `FolderFlag` integer, -- Folder Flag. The type of the folders. 0 = This is a folder, 1 = This is a document. Default: 0.
    `FileName` text, -- File Name. Uniquely identifying the record. Used to support a merge replication sample. File name of the document.
    `FileExtension` text, -- File Extension. File extension indicating the document type.
    `Revision` text, -- Revision number of the document.
    `ChangeNumber` integer, -- Change Number. Engineering change approval number.
    `Status` integer, -- Status of document processing. 1 = Pending approval, 2 = Approved, 3 = Obsolete.
    `DocumentSummary` text, -- Document Summary. Document abstract. if no document Summary: it means this document is private.
    `Document` blob, -- Complete document.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (DocumentNode)
);

CREATE TABLE CurrencyRate (
    `CurrencyRateID` integer, -- Currency Rate ID. The unique id number identifying the currency rate record.
    `CurrencyRateDate` datetime, -- Currency Rate Date. Date and time the exchange rate was obtained.
    `FromCurrencyCode` text, -- From Currency Code. Exchange rate was converted from this currency code.
    `ToCurrencyCode` text, -- Exchange rate was converted to this currency code.
    `AverageRate` real, -- Average Rate. Average exchange rate for the day.
    `EndOfDayRate` real, -- End Of Day Rate. Final exchange rate for the day.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CurrencyRateID)
);

CREATE TABLE ProductPhoto (
    `ProductPhotoID` integer, -- Product Photo ID. unique id number identifying the products.
    `ThumbNailPhoto` blob, -- Thumb Nail Photo. Small image of the product. The size of small images. "80x49 GIF image 3.07 kB". "3.07" represents the size of images, "GIF" refers to type of images like JPEG, GIF, etc.
    `ThumbnailPhotoFileName` text, -- Thumbnail Photo File Name. Small image file name.
    `LargePhoto` blob, -- large photo. Large image of the product.
    `LargePhotoFileName` text, -- Large Photo File Name. Large image file name. similar to ThumbNailPhoto.
    `ModifiedDate` datetime, -- Modified Date. Date and time the record was last updated.
    primary key (ProductPhotoID)
);

CREATE TABLE ProductReview (
    `ProductReviewID` integer, -- Product Review ID. The unique id numbers identifying product reviews.
    `ProductID` integer, -- Product ID. Product identification number.
    `ReviewerName` text, -- Reviewer Name. The name of reviewer.
    `ReviewDate` datetime, -- Review Date. Date review was submitted. Default: getdate().
    `EmailAddress` text, -- Email Address. Reviewer's e-mail address.
    `Rating` integer, -- Product rating given by the reviewer. Scale is 1 to 5 with 5 as the highest rating.
    `Comments` text, -- No description.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductReviewID),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE SalesOrderHeaderSalesReason (
    `SalesOrderID` integer, -- Sales Order ID. The id number of sales order.
    `SalesReasonID` integer, -- Sales Reason ID. The id number for sales reasons.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (SalesOrderID, SalesReasonID),
    foreign key (SalesOrderID) references SalesOrderHeader(SalesOrderID),
    foreign key (SalesReasonID) references SalesReason(SalesReasonID)
);

CREATE TABLE ContactType (
    `ContactTypeID` integer, -- Contact Type ID. The unique id number identifying the contact type ID.
    `Name` text, -- Contact type description.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ContactTypeID)
);

CREATE TABLE ProductDescription (
    `ProductDescriptionID` integer, -- Product Description ID. The unique id number identifying the product description.
    `Description` text, -- Description of the product.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductDescriptionID)
);

CREATE TABLE SalesPerson (
    `BusinessEntityID` integer, -- Business Entity ID. The unique id number identifying the business entity.
    `TerritoryID` integer, -- Territory ID. Territory currently assigned to.
    `SalesQuota` real, -- Sales Quota. Projected yearly sales.
    `Bonus` real, -- Bonus due if quota is met. if bonus is equal to 0, it means this salesperson doesn't meet quota. vice versa.
    `CommissionPct` real, -- Commission percentage. Commission percent received per sale.
    `SalesYTD` real, -- sales year to date. Sales total year to date.
    `SalesLastYear` real, -- Sales Last Year. Sales total of previous year.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID),
    foreign key (BusinessEntityID) references Employee(BusinessEntityID),
    foreign key (TerritoryID) references SalesTerritory(TerritoryID)
);

CREATE TABLE Password (
    `BusinessEntityID` integer, -- Business Entity ID. The unique id number identifying the person.
    `PasswordHash` text, -- Password Hash. Password for the e-mail account.
    `PasswordSalt` text, -- Password Salt. Random value concatenated with the password string before the password is hashed.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID),
    foreign key (BusinessEntityID) references Person(BusinessEntityID)
);

CREATE TABLE ProductVendor (
    `ProductID` integer, -- Product ID. The id number identifying the products.
    `BusinessEntityID` integer, -- Business Entity ID. The id number identifying the business entities.
    `AverageLeadTime` integer, -- Average Lead Time. The average span of time between placing an order with the vendor and receiving the purchased product. in days.
    `StandardPrice` real, -- Standard Price. The vendor's usual selling price.
    `LastReceiptCost` real, -- Last Receipt Cost. The selling price when last purchased. profit on net can be computed by LastReceiptCost - StandardPrice.
    `LastReceiptDate` datetime, -- Last Receipt Date. Date the product was last received by the vendor.
    `MinOrderQty` integer, -- Min Order Quantity. The maximum quantity that should be ordered.
    `MaxOrderQty` integer, -- Max Order Quantity. The minimum quantity that should be ordered.
    `OnOrderQty` integer, -- On Order Quantity. The quantity currently on order. if it's equal to 0, it means "out of stock".
    `UnitMeasureCode` text, -- Unit Measure Code. The product's unit of measure.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductID, BusinessEntityID),
    foreign key (ProductID) references Product(ProductID),
    foreign key (BusinessEntityID) references Vendor(BusinessEntityID),
    foreign key (UnitMeasureCode) references UnitMeasure(UnitMeasureCode)
);

CREATE TABLE ShoppingCartItem (
    `ShoppingCartItemID` integer, -- Shopping CartItem ID. The unique id number identifying the shopping cart item records.
    `ShoppingCartID` text, -- Shopping Cart ID. Shopping cart identification number.
    `Quantity` integer, -- Product quantity ordered. Default: 1.
    `ProductID` integer, -- Product ID. Product ordered.
    `DateCreated` datetime, -- Date Created. Date the time the record was created. Default: getdate().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ShoppingCartItemID),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE JobCandidate (
    `JobCandidateID` integer, -- Job Candidate ID. The unique id number identifying the job candidates.
    `BusinessEntityID` integer, -- Business Entity ID. Employee identification number if applicant was hired.
    `Resume` text, -- R�sum� in XML format.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (JobCandidateID)
);

CREATE TABLE ProductListPriceHistory (
    `ProductID` integer, -- Product ID. Product identification number.
    `StartDate` date, -- Start Date. List price start date.
    `EndDate` date, -- End Date. List price end date.
    `ListPrice` real, -- Product list price.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductID, StartDate),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE ScrapReason (
    `ScrapReasonID` integer, -- Scrap Reason ID. The unique number identifying for ScrapReason records.
    `Name` text, -- Failure description.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ScrapReasonID)
);

CREATE TABLE ProductSubcategory (
    `ProductSubcategoryID` integer, -- Product Subcategory ID. Unique id number identifying the subcategory of products.
    `ProductCategoryID` integer, -- Product Category ID. Product category identification number.
    `Name` text, -- Subcategory description.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductSubcategoryID),
    foreign key (ProductCategoryID) references ProductCategory(ProductCategoryID)
);

CREATE TABLE ShipMethod (
    `ShipMethodID` integer, -- Ship Method ID. The unique number for ShipMethod records.
    `Name` text, -- Shipping company name.
    `ShipBase` real, -- Ship Base. Minimum shipping charge.
    `ShipRate` real, -- Ship Rate. Shipping charge per pound.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ShipMethodID)
);

CREATE TABLE PurchaseOrderDetail (
    `PurchaseOrderID` integer, -- Purchase Order ID. Purchase order number.
    `PurchaseOrderDetailID` integer, -- Purchase Orde rDetail ID. Unique purchase detail order.
    `DueDate` datetime, -- Due Date. Date the product is expected to be received.
    `OrderQty` integer, -- Order Quantity. Quantity ordered.
    `ProductID` integer, -- Product ID. The id number identifying products.
    `UnitPrice` real, -- Unit Price. Vendor's selling price of a single product.
    `LineTotal` real, -- Line Total. Per product subtotal. Computed as OrderQty * UnitPrice. Computed: isnull([OrderQty]*[UnitPrice],(0.00)).
    `ReceivedQty` real, -- Quantity actually received from the vendor.
    `RejectedQty` real, -- No description.
    `StockedQty` real, -- No description.
    `ModifiedDate` real, -- No description.
    primary key (PurchaseOrderDetailID),
    foreign key (PurchaseOrderID) references PurchaseOrderHeader(PurchaseOrderID),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE ProductModelProductDescriptionCulture (
    `ProductModelID` integer, -- Product Model ID. The id number identifying the product model.
    `ProductDescriptionID` integer, -- Product Description ID. Product description identification number.
    `CultureID` text, -- Culture ID. Culture identification number.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductModelID, ProductDescriptionID, CultureID),
    foreign key (ProductModelID) references ProductModel(ProductModelID),
    foreign key (ProductDescriptionID) references ProductDescription(ProductDescriptionID),
    foreign key (CultureID) references Culture(CultureID)
);

CREATE TABLE SalesTerritoryHistory (
    `BusinessEntityID` integer, -- Business Entity ID. The sales representative.
    `TerritoryID` integer, -- Territory ID. Territory identification number.
    `StartDate` datetime, -- Start Date. Date the sales representive started work in the territory.
    `EndDate` datetime, -- End Date. Date the sales representative left work in the territory.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, StartDate, TerritoryID),
    foreign key (BusinessEntityID) references SalesPerson(BusinessEntityID),
    foreign key (TerritoryID) references SalesTerritory(TerritoryID)
);

CREATE TABLE Vendor (
    `BusinessEntityID` integer, -- The unique number identifying Vendor records.
    `AccountNumber` text, -- Vendor account (identification) number.
    `Name` text, -- Company name.
    `CreditRating` integer, -- Rating of credit. 1 = Superior, 2 = Excellent, 3 = Above average, 4 = Average, 5 = Below average. 1, 2, 3 can represent good credit, 4 is average credit, 5 is bad credit.
    `PreferredVendorStatus` integer, -- Preferred Vendor Status. 0 = Do not use if another vendor is available. 1 = Preferred over other vendors supplying the same product.
    `ActiveFlag` integer, -- Active Flag. Vendor URL. 0 = Vendor no longer used. 1 = Vendor is actively used. Default: 1.
    `PurchasingWebServiceURL` text, -- Purchasing Web Service URL.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID),
    foreign key (BusinessEntityID) references BusinessEntity(BusinessEntityID)
);

CREATE TABLE AddressType (
    `AddressTypeID` integer, -- address type id. Unique numbers identifying address type records. Primary key for AddressType records. Identity / Auto increment column.
    `Name` text, -- Address type description.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (AddressTypeID)
);

CREATE TABLE StateProvince (
    `StateProvinceID` integer, -- State Province ID. The unique number for StateProvince records.
    `StateProvinceCode` text, -- State Province Code. ISO standard state or province code.
    `CountryRegionCode` text, -- Country Region Code. ISO standard country or region code.
    `IsOnlyStateProvinceFlag` integer, -- Is Only State Province Flag. 0 = StateProvinceCode exists. 1 = StateProvinceCode unavailable, using CountryRegionCode. Default: 1. To ask whether the StateProvinceCode exists or not.
    `Name` text, -- State or province description.
    `TerritoryID` integer, -- Territory ID. ID of the territory in which the state or province is located.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (StateProvinceID)
);

CREATE TABLE WorkOrderRouting (
    `WorkOrderID` integer, -- The id number of work order.
    `ProductID` integer, -- The id number identifying products.
    `OperationSequence` integer, -- Operation Sequence. Indicates the manufacturing process sequence.
    `LocationID` integer, -- Location ID. Manufacturing location where the part is processed.
    `ScheduledStartDate` datetime, -- Scheduled Start Date. Planned manufacturing start date.
    `ScheduledEndDate` datetime, -- Scheduled End Date. Planned manufacturing end date.
    `ActualStartDate` datetime, -- Actual Start Date. Actual start date.
    `ActualEndDate` datetime, -- Actual end date.
    `ActualResourceHrs` real, -- Actual Resource Hours. Number of manufacturing hours used.
    `PlannedCost` real, -- Planned Cost. Estimated manufacturing cost.
    `ActualCost` real, -- Actual manufacturing cost.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (WorkOrderID, ProductID, OperationSequence),
    foreign key (WorkOrderID) references WorkOrder(WorkOrderID),
    foreign key (LocationID) references Location(LocationID)
);

CREATE TABLE PersonCreditCard (
    `BusinessEntityID` integer, -- Business Entity ID. The id number identifying the person.
    `CreditCardID` integer, -- Credit Card ID. Credit card identification number.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID, CreditCardID),
    foreign key (CreditCardID) references CreditCard(CreditCardID),
    foreign key (BusinessEntityID) references Person(BusinessEntityID)
);

CREATE TABLE Person (
    `BusinessEntityID` integer, -- Business Entity ID. The unique id number identifying the person.
    `PersonType` text, -- Person Type. The type of person. Primary type of person: SC = Store Contact, IN = Individual (retail) customer, SP = Sales person, EM = Employee (non-sales), VC = Vendor contact, GC = General contact.
    `NameStyle` integer, -- Name Style. Name Style. 0 = The data in FirstName and LastName are stored in western style (first name, last name) order. 1 = Eastern style (last name, first name) order. Default: 0.
    `Title` text, -- A courtesy title.
    `FirstName` text, -- First Name. First name of the person. Default: getdate().
    `MiddleName` text, -- Middle Name. Middle name or middle initial of the person.
    `LastName` text, -- Last Name. Last name of the person.
    `Suffix` text, -- Surname suffix.
    `EmailPromotion` integer, -- Email Promotion. Email Promotion. 0 = Contact does not wish to receive e-mail promotions, 1 = Contact does wish to receive e-mail promotions from AdventureWorks, 2 = Contact does wish to receive e-mail promotions from AdventureWorks and selected partners. Default: 0.
    `AdditionalContactInfo` text, -- Additional Contact Info. Additional contact information about the person stored in xml format.
    `Demographics` text, -- Personal information such as hobbies, and income collected from online shoppers. Used for sales analysis.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID),
    foreign key (BusinessEntityID) references BusinessEntity(BusinessEntityID)
);

CREATE TABLE SalesTerritory (
    `TerritoryID` integer, -- The unique id number for SalesTerritory records.
    `Name` text, -- Sales territory description.
    `CountryRegionCode` text, -- Country Region Code. ISO standard country or region code.
    `Group` text, -- Geographic area to which the sales territory belong.
    `SalesYTD` real, -- Sales Year to Date. Sales in the territory year to date.
    `SalesLastYear` real, -- Sales Last Year. Sales in the territory the previous year.
    `CostYTD` real, -- Cost Year to Date. Business costs in the territory year to date.
    `CostLastYear` real, -- Cost Last Year. Business costs in the territory the previous year.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (TerritoryID),
    foreign key (CountryRegionCode) references CountryRegion(CountryRegionCode)
);

CREATE TABLE SpecialOfferProduct (
    `SpecialOfferID` integer, -- Special Offer ID. The id for SpecialOfferProduct records.
    `ProductID` integer, -- Product ID. Product identification number.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (SpecialOfferID, ProductID),
    foreign key (SpecialOfferID) references SpecialOffer(SpecialOfferID),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE SpecialOffer (
    `SpecialOfferID` integer, -- Special Offer ID. The unique id number identifying the special offer.
    `Description` text, -- Discount description.
    `DiscountPct` real, -- Discount precentage. Discount percentage.
    `Type` text, -- Discount type category.
    `Category` text, -- Group the discount applies to such as Reseller or Customer.
    `StartDate` datetime, -- Start Date. Discount start date.
    `EndDate` datetime, -- End Date. Discount end date. promotion date = EndDate - StartDate.
    `MinQty` integer, -- Min Quality. Minimum discount percent allowed.
    `MaxQty` integer, -- Max Quality. Maximum discount percent allowed.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (SpecialOfferID)
);

CREATE TABLE Currency (
    `CurrencyCode` text, -- Currency Code. The unique id string identifying the currency.
    `Name` text, -- Currency name.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CurrencyCode)
);

CREATE TABLE ProductInventory (
    `ProductID` integer, -- Product ID. Product identification number.
    `LocationID` integer, -- Location ID. Inventory location identification number. Document identification number.
    `Shelf` text, -- Storage compartment within an inventory location.
    `Bin` integer, -- Storage container on a shelf in an inventory location.
    `Quantity` integer, -- Quantity of products in the inventory location. Default: 0.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductID, LocationID),
    foreign key (ProductID) references Product(ProductID),
    foreign key (LocationID) references Location(LocationID)
);

CREATE TABLE Store (
    `BusinessEntityID` integer, -- Business Entity ID. The unique number identifying business entity.
    `Name` text, -- Name of the store.
    `SalesPersonID` integer, -- Sales Person ID. ID of the sales person assigned to the customer.
    `Demographics` text, -- Demographic information about the store such as the number of employees, annual sales and store type.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (BusinessEntityID),
    foreign key (BusinessEntityID) references BusinessEntity(BusinessEntityID),
    foreign key (SalesPersonID) references SalesPerson(BusinessEntityID)
);

CREATE TABLE SalesOrderHeader (
    `SalesOrderID` integer, -- Sales Order ID. The id number of sales order.
    `RevisionNumber` integer, -- No description.
    `OrderDate` datetime, -- Order Date.
    `DueDate` datetime, -- No description.
    `ShipDate` datetime, -- Ship Date. Estimated shipment date from the vendor.
    `Status` integer, -- Order current status. 1 = Pending; 2 = Approved; 3 = Rejected; 4 = Complete Default: 1.
    `OnlineOrderFlag` integer, -- No description.
    `SalesOrderNumber` text, -- No description.
    `PurchaseOrderNumber` text, -- No description.
    `AccountNumber` text, -- No description.
    `CustomerID` integer, -- No description.
    `SalesPersonID` integer, -- No description.
    `TerritoryID` integer, -- No description.
    `BillToAddressID` integer, -- No description.
    `ShipToAddressID` integer, -- No description.
    `ShipMethodID` integer, -- Ship Method ID. Shipping method.
    `CreditCardID` integer, -- No description.
    `CreditCardApprovalCode` text, -- No description.
    `CurrencyRateID` integer, -- No description.
    `SubTotal` real, -- No description.
    `TaxAmt` real, -- Tax Amount. Tax amount.
    `Freight` real, -- Shipping cost.
    `TotalDue` real, -- Total Due. Total due to vendor.
    `Comment` text, -- No description.
    `rowguid` text, -- No description.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (SalesOrderID)
);

CREATE TABLE CreditCard (
    `CreditCardID` integer, -- Credit Card ID. The unique id number identifying the credit card.
    `CardType` text, -- Card Type. Credit card name.
    `CardNumber` text, -- Card Number. Credit card number.
    `ExpMonth` integer, -- Expiration Month. Credit card expiration month.
    `ExpYear` integer, -- Expiration Year. Credit card expiration year.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CreditCardID)
);

CREATE TABLE TransactionHistoryArchive (
    `TransactionID` integer, -- Transaction ID. The unique id number identifying TransactionHistory records.
    `ProductID` integer, -- Product ID. Product identification number.
    `ReferenceOrderID` integer, -- Reference Order ID. Purchase order, sales order, or work order identification number.
    `ReferenceOrderLineID` integer, -- Reference Order Line ID. Line number associated with the purchase order, sales order, or work order.
    `TransactionDate` datetime, -- Transaction Date. Date and time of the transaction.
    `TransactionType` text, -- Transaction Type. Type of transaction records. W = WorkOrder, S = SalesOrder, P = PurchaseOrder.
    `Quantity` integer, -- Product quantity.
    `ActualCost` real, -- Actual Cost. Product cost.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (TransactionID)
);

CREATE TABLE TransactionHistory (
    `TransactionID` integer, -- Transaction ID. The unique id number identifying TransactionHistory records.
    `ProductID` integer, -- Product ID. Product identification number.
    `ReferenceOrderID` integer, -- Reference Order ID. Purchase order, sales order, or work order identification number.
    `ReferenceOrderLineID` integer, -- Reference Order Line ID. Line number associated with the purchase order, sales order, or work order.
    `TransactionDate` datetime, -- Transaction Date. Date and time of the transaction.
    `TransactionType` text, -- Transaction Type. Type of transaction records. W = WorkOrder, S = SalesOrder, P = PurchaseOrder.
    `Quantity` integer, -- Product quantity.
    `ActualCost` real, -- Actual Cost. Product cost.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (TransactionID),
    foreign key (ProductID) references Product(ProductID)
);

CREATE TABLE Customer (
    `CustomerID` integer, -- Customer ID. The unique id number identifying the customer.
    `PersonID` integer, -- Person ID. The id number identifying the person.
    `StoreID` integer, -- Store ID. The id number identifying the store / bussiness entity.
    `TerritoryID` integer, -- Territory ID. ID of the territory in which the customer is located.
    `AccountNumber` text, -- Account Number. Unique number identifying the customer assigned by the accounting system. Computed: isnull('AW'+[ufnLeadingZeros]([CustomerID]),'').
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CustomerID),
    foreign key (PersonID) references Person(BusinessEntityID),
    foreign key (TerritoryID) references SalesTerritory(TerritoryID),
    foreign key (StoreID) references Store(BusinessEntityID)
);

CREATE TABLE ProductModel (
    `ProductModelID` integer, -- Product Model ID. The unique id number identifying the product model.
    `Name` text, -- Product model description.
    `CatalogDescription` text, -- Catalog Description. Detailed product catalog information in xml format.
    `Instructions` text, -- Manufacturing instructions in xml format.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (ProductModelID)
);

CREATE TABLE Address (
    `AddressID` integer, -- address id. Unique id number identifying the address. Primary key for Address records. Identity / Auto increment column.
    `AddressLine1` text, -- address line 1. First street address line.
    `AddressLine2` text, -- address line 2. Second street address line. 1. total address = (AddressLine1+AddressLine2). 2. if AddressLine2 is not null, it means the address is too long.
    `City` text, -- Name of the city.
    `StateProvinceID` integer, -- state province id. Identification number for the state or province.
    `PostalCode` text, -- postal code. Postal code for the street address.
    `SpatialLocation` text, -- spatial location. Latitude and longitude of this address.
    `rowguid` text, -- Uniquely identifying the record. Used to support a merge replication sample. Default: newid().
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (AddressID)
);

CREATE TABLE Culture (
    `CultureID` text, -- Culture ID. The unique id string identifying the language in which AdventrueWorks data is stored.
    `Name` text, -- Name of the language.
    `ModifiedDate` datetime, -- modified date. Date and time the record was last updated. Default: getdate().
    primary key (CultureID)
);

