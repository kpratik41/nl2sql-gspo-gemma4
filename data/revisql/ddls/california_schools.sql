CREATE TABLE schools (
    `CDSCode` text, -- No description.
    `NCESDist` text, -- National Center for Educational Statistics school district identification number. This field represents the 7-digit National Center for Educational Statistics (NCES) school district identification number. The first 2 digits identify the state and the last 5 digits identify the school district. Combined, they make a unique 7-digit ID for each school district.
    `NCESSchool` text, -- National Center for Educational Statistics school identification number. This field represents the 5-digit NCES school identification number. The NCESSchool combined with the NCESDist form a unique 12-digit ID for each school.
    `StatusType` text, -- This field identifies the status of the district. Definitions of the valid status types are listed below:. · Active: The district is in operation and providing instructional services. · Closed: The district is not in operation and no longer providing instructional services. · Merged: The district has combined with another district or districts. · Pending: The district has not opened for operation and instructional services yet, but plans to open within the next 9–12 months.
    `County` text, -- County name.
    `District` text, -- District.
    `School` text, -- School. If this field is NULL, then the record is for a district, not a school.
    `Street` text, -- Street.
    `StreetAbr` text, -- street address. The abbreviated street address of the school, district, or administrative authority’s physical location. The abbreviated street address of the school, district, or administrative authority’s physical location. Note: Some records (primarily records of closed or retired schools) may not have data in this field.
    `City` text, -- City.
    `Zip` text, -- Zip.
    `State` text, -- State.
    `MailStreet` text, -- MailStreet. The unabbreviated mailing address of the school, district, or administrative authority. Note: 1) Some entities (primarily closed or retired schools) may not have data in this field; 2) Many active entities have not provided a mailing street address. For your convenience we have filled the unpopulated MailStreet cells with Street data.
    `MailStrAbr` text, -- mailing street address. the abbreviated mailing street address of the school, district, or administrative authority.Note: Many active entities have not provided a mailing street address. For your convenience we have filled the unpopulated MailStrAbr cells with StreetAbr data.
    `MailCity` text, -- mailing city. The city associated with the mailing address of the school, district, or administrative authority. Note: Many entities have not provided a mailing address city. For your convenience we have filled the unpopulated MailCity cells with City data.
    `MailZip` text, -- mailing zip. The zip code associated with the mailing address of the school, district, or administrative authority. Note: Many entities have not provided a mailing address zip code. For your convenience we have filled the unpopulated MailZip cells with Zip data.
    `MailState` text, -- mailing state. The state within the mailing address. For your convenience we have filled the unpopulated MailState cells with State data.
    `Phone` text, -- Phone.
    `Ext` text, -- extension. The phone number extension of the school, district, or administrative authority.
    `Website` text, -- The website address of the school, district, or administrative authority.
    `OpenDate` date, -- The date the school opened.
    `ClosedDate` date, -- The date the school closed.
    `Charter` integer, -- This field identifies a charter school. The field is coded as follows:. · 1 = The school is a charter. · 0 = The school is not a charter.
    `CharterNum` text, -- The charter school number,. 4-digit number assigned to a charter school.
    `FundingType` text, -- Indicates the charter school funding type. Values are as follows:. · Not in CS (California School) funding model. · Locally funded. · Directly funded.
    `DOC` text, -- District Ownership Code. The District Ownership Code (DOC) is the numeric code used to identify the category of the Administrative Authority. • 00 - County Office of Education. • 02 – State Board of Education. • 03 – Statewide Benefit Charter. • 31 – State Special Schools. • 34 – Non-school Location*. • 52 – Elementary School District. • 54 – Unified School District. • 56 – High School District. • 98 – Regional Occupational Center/Program (ROC/P). *Only the California Education Authority has been included in the non-school location category.
    `DOCType` text, -- The District Ownership Code Type. The District Ownership Code Type is the text description of the DOC category. (See text values in DOC field description above).
    `SOC` text, -- School Ownership Code. The School Ownership Code is a numeric code used to identify the type of school. • 08 - Preschool. • 09 – Special Education Schools (Public). • 11 – Youth Authority Facilities (CEA). • 13 – Opportunity Schools. • 14 – Juvenile Court Schools. • 15 – Other County or District Programs. • 31 – State Special Schools. • 60 – Elementary School (Public). • 61 – Elementary School in 1 School District (Public). • 62 – Intermediate/Middle Schools (Public). • 63 – Alternative Schools of Choice. • 64 – Junior High Schools (Public). • 65 – K-12 Schools (Public). • 66 – High Schools (Public). • 67 – High Schools in 1 School District (Public). • 68 – Continuation High Schools. • 69 – District Community Day Schools. • 70 – Adult Education Centers. • 98 – Regional Occupational Center/Program (ROC/P).
    `SOCType` text, -- School Ownership Code Type. The School Ownership Code Type is the text description of the type of school.
    `EdOpsCode` text, -- Education Option Code. The Education Option Code is a short text description of the type of education offered. • ALTSOC – Alternative School of Choice. • COMM – County Community School. • COMMDAY – Community Day School. • CON – Continuation School. • JUV – Juvenile Court School. • OPP – Opportunity School. • YTH – Youth Authority School. • SSS – State Special School. • SPEC – Special Education School. • TRAD – Traditional. • ROP – Regional Occupational Program. • HOMHOS – Home and Hospital. • SPECON – District Consortia Special Education School.
    `EdOpsName` text, -- Educational Option Name. The Educational Option Name is the long text description of the type of education being offered.
    `EILCode` text, -- Educational Instruction Level Code. The Educational Instruction Level Code is a short text description of the institution's type relative to the grade range served. • A – Adult. • ELEM – Elementary. • ELEMHIGH – Elementary-High Combination. • HS – High School. • INTMIDJR – Intermediate/Middle/Junior High. • PS – Preschool. • UG – Ungraded.
    `EILName` text, -- Educational Instruction Level Name. The Educational Instruction Level Name is the long text description of the institution’s type relative to the grade range served.
    `GSoffered` text, -- grade span offered. The grade span offered is the lowest grade and the highest grade offered or supported by the school, district, or administrative authority. This field might differ from the grade span served as reported in the most recent certified California Longitudinal Pupil Achievement (CALPADS) Fall 1 data collection. For example XYZ School might display the following data:. GSoffered = P–Adult. GSserved = K–12.
    `GSserved` text, -- grade span served.. It is the lowest grade and the highest grade of student enrollment as reported in the most recent certified CALPADS Fall 1 data collection. Only K–12 enrollment is reported through CALPADS. This field may differ from the grade span offered. 1. Only K–12 enrollment is reported through CALPADS. 2. Note: Special programs at independent study, alternative education, and special education schools will often exceed the typical grade span for schools of that type.
    `Virtual` text, -- This field identifies the type of virtual instruction offered by the school. Virtual instruction is instruction in which students and teachers are separated by time and/or location, and interaction occurs via computers and/or telecommunications technologies. The field is coded as follows:. · F = Exclusively Virtual – The school has no physical building where students meet with each other or with teachers, all instruction is virtual. · V = Primarily Virtual – The school focuses on a systematic program of virtual instruction but includes some physical meetings among students or with teachers. · C = Primarily Classroom – The school offers virtual courses but virtual instruction is not the primary means of instruction. · N = Not Virtual – The school does not offer any virtual instruction. · P = Partial Virtual – The school offers some, but not all, instruction through virtual instruction. Note: This value was retired and replaced with the Primarily Virtual and Primarily Classroom values beginning with the 2016–17 school year.
    `Magnet` integer, -- This field identifies whether a school is a magnet school and/or provides a magnet program. The field is coded as follows:. · 1 = Magnet - The school is a magnet school and/or offers a magnet program. · 0 = Not Magnet - The school is not a magnet school and/or does not offer a magnet program. Note: Preschools and adult education centers do not contain a magnet school indicator.
    `Latitude` real, -- The angular distance (expressed in degrees) between the location of the school, district, or administrative authority and the equator measured north to south.
    `Longitude` real, -- The angular distance (expressed in degrees) between the location of the school, district, or administrative authority and the prime meridian (Greenwich, England) measured from west to east.
    `AdmFName1` text, -- administrator's first name. The superintendent’s or principal’s first name. Only active and pending districts and schools will display administrator information, if applicable.
    `AdmLName1` text, -- administrator's last name. The superintendent’s or principal’s last name. Only active and pending districts and schools will display administrator information, if applicable.
    `AdmEmail1` text, -- administrator's email address. The superintendent’s or principal’s email address. Only active and pending districts and schools will display administrator information, if applicable.
    `AdmFName2` text, -- SAME as 1.
    `AdmLName2` text, -- No description.
    `AdmEmail2` text, -- No description.
    `AdmFName3` text, -- not useful.
    `AdmLName3` text, -- not useful.
    `AdmEmail3` text, -- not useful.
    `LastUpdate` date, -- when is this record updated last time.
    primary key (CDSCode)
);

CREATE TABLE satscores (
    `cds` text, -- California Department Schools.
    `rtype` text, -- type of record. The value is either 'D' or 'S'. 'D' means the record is for a district, and 'S' means the record is for a school.
    `sname` text, -- school name. If sname is null, then this record is for a district.
    `dname` text, -- district name. district segment.
    `cname` text, -- county name.
    `enroll12` integer, -- enrollment (1st-12nd grade).
    `NumTstTakr` integer, -- Number of Test Takers. Number of Test Takers in this school. number of test takers in each school.
    `AvgScrRead` integer, -- average scores in Reading.
    `AvgScrMath` integer, -- average scores in Math.
    `AvgScrWrite` integer, -- average scores in writing.
    `NumGE1500` integer, -- Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500. Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500. Excellence Rate = NumGE1500 / NumTstTakr.
    primary key (cds),
    foreign key (cds) references schools (CDSCode)
);

CREATE TABLE frpm (
    `CDSCode` integer, -- CDSCode.
    `Academic Year` integer , -- Academic Year.
    `County Code` integer, -- County Code.
    `District Code` integer, -- District Code.
    `School Code` integer, -- School Code.
    `County Name` text, -- County Code.
    `District Name` text, -- District Name.
    `School Name` text, -- School Name.
    `District Type` text, -- District Type.
    `School Type` text, -- School Type.
    `Educational Option Type` text, -- Educational Option Type.
    `NSLP Provision Status` text, -- NSLP Provision Status.
    `Charter School (Y/N)` integer, -- Charter School (Y/N). 0: N; 1: Y.
    `Charter School Number` text, -- Charter School Number.
    `Charter Funding Type` text, -- Charter Funding Type.
    `IRC` integer, -- Not useful.
    `Low Grade` text, -- Low Grade.
    `High Grade` text, -- High Grade.
    `Enrollment (K-12)` real, -- Enrollment (K-12). K-12: kindergarten - 12th grade.
    `Free Meal Count (K-12)` real, -- Free Meal Count (K-12). eligible free rate = Free Meal Count / Enrollment.
    `Percent (%) Eligible Free (K-12)` real, -- No description.
    `FRPM Count (K-12)` real, -- Free or Reduced Price Meal Count (K-12). eligible FRPM rate = FRPM / Enrollment.
    `Percent (%) Eligible FRPM (K-12)` real, -- No description.
    `Enrollment (Ages 5-17)` real, -- Enrollment (Ages 5-17).
    `Free Meal Count (Ages 5-17)` real, -- Free Meal Count (Ages 5-17). eligible free rate = Free Meal Count / Enrollment.
    `Percent (%) Eligible Free (Ages 5-17)` real, -- No description.
    `FRPM Count (Ages 5-17)` real, -- No description.
    `Percent (%) Eligible FRPM (Ages 5-17)` real, -- No description.
    `2013-14 CALPADS Fall 1 Certification Status` integer, -- 2013-14 CALPADS Fall 1 Certification Status.
    primary key (CDSCode),
    foreign key (CDSCode) references schools (CDSCode)
);

