CREATE TABLE Agents (
    AgentID int NOT NULL PRIMARY KEY, -- example: [1, 2, 3]
    AgtFirstName nvarchar (25) NULL, -- example: ['William', 'Scott', 'Carol']
    AgtLastName nvarchar (25) NULL, -- example: ['Thompson', 'Bishop', 'Viescas']
    AgtStreetAddress nvarchar (50) NULL, -- example: ['122 Spring River Drive', '66 Spring Valley Drive', '667 Red River Road']
    AgtCity nvarchar (30) NULL, -- example: ['Redmond', 'Seattle', 'Bellevue']
    AgtState nvarchar (2) NULL, -- example: ['WA']
    AgtZipCode nvarchar (10) NULL, -- example: ['98006', '98033', '98052']
    AgtPhoneNumber nvarchar (15) NULL, -- example: ['555-2681', '555-2666', '555-2571']
    DateHired date NULL, -- example: ['1997-05-15', '1998-02-05', '1997-11-19']
    Salary decimal(15, 2) NULL DEFAULT 0, -- example: [35000, 27000, 30000]
    CommissionRate float(24) NULL DEFAULT 0 -- example: [0.04, 0.05, 0.055]
)

CREATE TABLE Customers (
    CustomerID int NOT NULL PRIMARY KEY, -- example: [10001, 10002, 10003]
    CustFirstName nvarchar (25) NULL, -- example: ['Doris', 'Deb', 'Peter']
    CustLastName nvarchar (25) NULL, -- example: ['Hartwig', 'Waldal', 'Brehm']
    CustStreetAddress nvarchar (50) NULL, -- example: ['4726 - 11th Ave. N.E.', '908 W. Capital Way', '722 Moss Bay Blvd.']
    CustCity nvarchar (30) NULL, -- example: ['Seattle', 'Tacoma', 'Kirkland']
    CustState nvarchar (2) NULL, -- example: ['WA']
    CustZipCode nvarchar (10) NULL, -- example: ['98002', '98006', '98033']
    CustPhoneNumber nvarchar (15) NULL -- example: ['555-2671', '555-2496', '555-2501']
)

CREATE TABLE Engagements (
    EngagementNumber int NOT NULL PRIMARY KEY DEFAULT 0, -- example: [2, 3, 4]
    StartDate date NULL, -- example: ['2017-09-02', '2017-09-11', '2017-09-12']
    EndDate date NULL, -- example: ['2017-09-06', '2017-09-16', '2017-09-18']
    StartTime time NULL, -- example: ['13:00:00', '20:00:00', '16:00:00']
    StopTime time NULL, -- example: ['15:00:00', '00:00:00', '19:00:00']
    ContractPrice decimal(15, 2) NULL DEFAULT 0, -- example: [200, 590, 470]
    CustomerID int NULL DEFAULT 0, -- example: [10001, 10002, 10003]
    AgentID int NULL DEFAULT 0, -- example: [1, 2, 3]
    EntertainerID int NULL DEFAULT 0, -- example: [1001, 1002, 1003]
    FOREIGN KEY (AgentID) REFERENCES Agents(AgentID),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (EntertainerID) REFERENCES Entertainers(EntertainerID)
)

CREATE TABLE Entertainer_Members (
    EntertainerID int NOT NULL, -- example: [1001, 1002, 1003]
    MemberID int NOT NULL DEFAULT 0, -- example: [101, 102, 103]
    Status smallint NULL DEFAULT 0, -- example: [1, 2]
    PRIMARY KEY (EntertainerID, MemberID),
    FOREIGN KEY (EntertainerID) REFERENCES Entertainers(EntertainerID),
    FOREIGN KEY (MemberID) REFERENCES Members(MemberID)
)

CREATE TABLE Entertainer_Styles (
    EntertainerID int NOT NULL, -- example: [1001, 1002, 1003]
    StyleID smallint NOT NULL DEFAULT 0, -- example: [3, 4, 6]
    StyleStrength smallint NOT NULL, -- example: [2, 1, 3]
    PRIMARY KEY (EntertainerID, StyleID),
    FOREIGN KEY (EntertainerID) REFERENCES Entertainers(EntertainerID),
    FOREIGN KEY (StyleID) REFERENCES Musical_Styles(StyleID)
)

CREATE TABLE Entertainers (
    EntertainerID int NOT NULL PRIMARY KEY, -- example: [1001, 1002, 1003]
    EntStageName nvarchar (50) NULL, -- example: ['Carol Peacock Trio', 'Topazz', 'JV & the Deep Six']
    EntSSN nvarchar (12) NULL, -- example: ['888-90-1121', '888-50-1061', '888-18-1013']
    EntStreetAddress nvarchar (50) NULL, -- example: ['4110 Old Redmond Rd.', '16 Maple Lane', '15127 NE 24th, #383']
    EntCity nvarchar (30) NULL, -- example: ['Redmond', 'Auburn', 'Bellevue']
    EntState nvarchar (2) NULL, -- example: ['WA']
    EntZipCode nvarchar (10) NULL, -- example: ['98002', '98005', '98006']
    EntPhoneNumber nvarchar (15) NULL, -- example: ['555-2691', '555-2591', '555-2511']
    EntWebPage nvarchar (50) NULL, -- example: ['www.cptrio.com', 'www.topazz.com', 'www.jvd6.com']
    EntEMailAddress nvarchar (50) NULL, -- example: ['carolp@cptrio.com', 'jv@myspring.com', 'mikeh@moderndance.com']
    DateEntered date NULL -- example: ['1997-05-24', '1996-02-14', '1998-03-18']
)

CREATE TABLE Members (
    MemberID int NOT NULL PRIMARY KEY DEFAULT 0, -- example: [101, 102, 103]
    MbrFirstName nvarchar (25) NULL, -- example: ['David', 'Suzanne', 'Gary']
    MbrLastName nvarchar (25) NULL, -- example: ['Hamilton', 'Viescas', 'Hallmark']
    MbrPhoneNumber nvarchar (15) NULL, -- example: ['555-2701', '555-2686', '555-2676']
    Gender nvarchar (2) NULL -- example: ['M', 'F']
)

CREATE TABLE Musical_Preferences (
    CustomerID int NOT NULL DEFAULT 0, -- example: [10001, 10002, 10003]
    StyleID smallint NOT NULL DEFAULT 0, -- example: [1, 3, 4]
    PreferenceSeq smallint NOT NULL, -- example: [2, 1, 3]
    PRIMARY KEY (CustomerID, StyleID),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (StyleID) REFERENCES Musical_Styles(StyleID)
)

CREATE TABLE Musical_Styles (
    StyleID smallint NOT NULL PRIMARY KEY DEFAULT 0, -- example: [1, 2, 3]
    StyleName nvarchar (75) NULL -- example: ['40''s Ballroom Music', '50''s Music', '60''s Music']
)

CREATE TABLE ztblDays (
    DateField date NOT NULL PRIMARY KEY -- example: ['2017-01-01', '2017-01-02', '2017-01-03']
)

CREATE TABLE ztblMonths (
    MonthYear nvarchar (15) NULL, -- example: ['April 2017', 'April 2018', 'April 2019']
    YearNumber smallint NOT NULL, -- example: [2017, 2018, 2019]
    MonthNumber smallint NOT NULL, -- example: [1, 2, 3]
    MonthStart date NULL, -- example: ['2017-01-01', '2017-02-01', '2017-03-01']
    MonthEnd date NULL, -- example: ['2017-01-31', '2017-02-28', '2017-03-31']
    January smallint NULL DEFAULT 0, -- example: [1, 0]
    February smallint NULL DEFAULT 0, -- example: [0, 1]
    March smallint NULL DEFAULT 0, -- example: [0, 1]
    April smallint NULL DEFAULT 0, -- example: [0, 1]
    May smallint NULL DEFAULT 0, -- example: [0, 1]
    June smallint NULL DEFAULT 0, -- example: [0, 1]
    July smallint NULL DEFAULT 0, -- example: [0, 1]
    August smallint NULL DEFAULT 0, -- example: [0, 1]
    September smallint NULL DEFAULT 0, -- example: [0, 1]
    October smallint NULL DEFAULT 0, -- example: [0, 1]
    November smallint NULL DEFAULT 0, -- example: [0, 1]
    December smallint NULL DEFAULT 0, -- example: [0, 1]
    PRIMARY KEY (YearNumber, MonthNumber)
)

CREATE TABLE ztblSkipLabels (
    LabelCount int NOT NULL PRIMARY KEY -- example: [1, 2, 3]
)

CREATE TABLE ztblWeeks (
    WeekStart date NOT NULL PRIMARY KEY, -- example: ['2017-01-01', '2017-01-08', '2017-01-15']
    WeekEnd date NULL -- example: ['2017-01-07', '2017-01-14', '2017-01-21']
)

