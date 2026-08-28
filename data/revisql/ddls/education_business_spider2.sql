CREATE TABLE "hardware_dim_customer" (
"customer_code" INTEGER, -- example: [70002017, 70002018, 70003181]
  "customer" TEXT, -- example: ['Atliq Exclusive', 'Atliq e Store', 'Neptune']
  "platform" TEXT, -- example: ['Brick & Mortar', 'E-Commerce']
  "channel" TEXT, -- example: ['Direct', 'Distributor', 'Retailer']
  "market" TEXT, -- example: ['India', 'Indonesia', 'Japan']
  "sub_zone" TEXT, -- example: ['India', 'ROA', 'ANZ']
  "region" TEXT -- example: ['APAC', 'EU', 'LATAM']
)

CREATE TABLE "hardware_fact_pre_invoice_deductions" (
"customer_code" INTEGER, -- example: [70002017, 70002018, 70003181]
  "fiscal_year" INTEGER, -- example: [2020, 2021]
  "pre_invoice_discount_pct" REAL -- example: [0.0735, 0.0703, 0.2255]
)

CREATE TABLE "web_sales_reps" (
"id" INTEGER, -- example: [321500, 321510, 321520]
  "name" TEXT, -- example: ['Samuel Racine', 'Eugena Esser', 'Michel Averette']
  "region_id" INTEGER -- example: [1, 2, 3]
)

CREATE TABLE "hardware_dim_product" (
"product_code" TEXT, -- example: ['A0118150101', 'A0118150102', 'A0118150103']
  "division" TEXT, -- example: ['P & A', 'PC', 'N & S']
  "segment" TEXT, -- example: ['Peripherals', 'Accessories', 'Notebook']
  "category" TEXT, -- example: ['Internal HDD', 'Graphic Card', 'Processors']
  "product" TEXT, -- example: ['AQ Dracula HDD – 3.5 Inch SATA 6 Gb/s 5400 RPM 256 MB Cache', 'AQ WereWolf NAS Internal Hard Drive HDD – 8.89 cm', 'AQ Zion Saga']
  "variant" TEXT -- example: ['Standard', 'Plus', 'Premium']
)

CREATE TABLE "web_orders" (
"id" INTEGER, -- example: [1, 2, 3]
  "account_id" INTEGER, -- example: [1001, 1011, 1021]
  "occurred_at" TEXT, -- example: ['2015-10-06T17:31:14.000Z', '2015-11-05T03:34:33.000Z', '2015-12-04T04:21:55.000Z']
  "standard_qty" INTEGER, -- example: [123, 190, 85]
  "gloss_qty" INTEGER, -- example: [22, 41, 47]
  "poster_qty" INTEGER, -- example: [24, 57, 0]
  "total" INTEGER, -- example: [169, 288, 132]
  "standard_amt_usd" REAL, -- example: [613.77, 948.1, 424.15]
  "gloss_amt_usd" REAL, -- example: [164.78, 307.09, 352.03]
  "poster_amt_usd" REAL, -- example: [194.88, 462.84, 0.0]
  "total_amt_usd" REAL -- example: [973.43, 1718.03, 776.18]
)

CREATE TABLE "StaffHours" (
"StaffMember" TEXT, -- example: ['B', 'A', 'C']
  "EventDate" TEXT, -- example: ['2013-02-01', '2013-01-15', '2013-03-01']
  "EventTime" TEXT, -- example: ['09:00', '08:00', '07:45']
  "EventType" TEXT -- example: ['Enter', 'Exit']
)

CREATE TABLE "university_enrollment" (
"OfferNo" INTEGER, -- example: [1234, 4321, 5555]
  "StdNo" INTEGER, -- example: [123456789, 234567890, 345678901]
  "EnrGrade" REAL -- example: [3.3, 3.5, 3.2]
)

CREATE TABLE "university_faculty" (
"FacNo" INTEGER, -- example: [98765432, 543210987, 654321098]
  "FacFirstName" TEXT, -- example: ['LEONARD', 'VICTORIA', 'NICKI']
  "FacLastName" TEXT, -- example: ['VINCE', 'EMMANUEL', 'FIBON']
  "FacCity" TEXT, -- example: ['SEATTLE', 'BOTHELL', 'BELLEVUE']
  "FacState" TEXT, -- example: ['WA']
  "FacDept" TEXT, -- example: ['MS', 'FIN', 'CS']
  "FacRank" TEXT, -- example: ['ASST', 'PROF', 'ASSC']
  "FacSalary" INTEGER, -- example: [35000, 120000, 70000]
  "FacSupervisor" REAL, -- example: [654321098.0, 543210987.0, 765432109.0]
  "FacHireDate" TEXT, -- example: ['1997-04-10', '1998-04-15', '1996-05-01']
  "FacZipCode" TEXT -- example: ['98111-9921', '98011-2242', '98121-0094']
)

CREATE TABLE "university_student" (
"StdNo" INTEGER, -- example: [123456789, 124567890, 234567890]
  "StdFirstName" TEXT, -- example: ['HOMER', 'BOB', 'CANDY']
  "StdLastName" TEXT, -- example: ['WELLS', 'NORBERT', 'KENDALL']
  "StdCity" TEXT, -- example: ['SEATTLE', 'BOTHELL', 'TACOMA']
  "StdState" TEXT, -- example: ['WA']
  "StdZip" TEXT, -- example: ['98121-1111', '98011-2121', '99042-3321']
  "StdMajor" TEXT, -- example: ['IS', 'FIN', 'ACCT']
  "StdClass" TEXT, -- example: ['FR', 'JR', 'SR']
  "StdGPA" REAL -- example: [3.0, 2.7, 3.5]
)

CREATE TABLE "university_offering" (
"OfferNo" INTEGER, -- example: [1111, 1234, 2222]
  "CourseNo" TEXT, -- example: ['IS320', 'IS460', 'FIN300']
  "OffTerm" TEXT, -- example: ['SUMMER', 'FALL', 'SPRING']
  "OffYear" INTEGER, -- example: [2010, 2009]
  "OffLocation" TEXT, -- example: ['BLM302', 'BLM412', 'BLM214']
  "OffTime" TEXT, -- example: ['10:30 AM', '1:30 PM', '8:30 AM']
  "FacNo" REAL, -- example: [98765432.0, 543210987.0, 765432109.0]
  "OffDays" TEXT -- example: ['MW', 'TTH']
)

CREATE TABLE "web_accounts" (
"id" INTEGER, -- example: [1001, 1011, 1021]
  "name" TEXT, -- example: ['Walmart', 'Exxon Mobil', 'Apple']
  "website" TEXT, -- example: ['www.walmart.com', 'www.exxonmobil.com', 'www.apple.com']
  "lat" REAL, -- example: [40.23849561, 41.1691563, 42.29049481]
  "long" REAL, -- example: [-75.10329704, -73.84937379, -76.08400942]
  "primary_poc" TEXT, -- example: ['Tamara Tuma', 'Sung Shields', 'Jodee Lupo']
  "sales_rep_id" INTEGER -- example: [321500, 321510, 321520]
)

CREATE TABLE "web_events" (
"id" INTEGER, -- example: [1, 2, 3]
  "account_id" INTEGER, -- example: [1001, 1011, 1021]
  "occurred_at" TEXT, -- example: ['2015-10-06T17:13:58.000Z', '2015-11-05T03:08:26.000Z', '2015-12-04T03:57:24.000Z']
  "channel" TEXT -- example: ['direct', 'facebook', 'organic']
)

CREATE TABLE "SalaryDataset" (
"index" INTEGER, -- example: [0, 1, 2]
  "CompanyName" TEXT, -- example: ['Mu Sigma', 'IBM', 'Tata Consultancy Services']
  "JobTitle" TEXT, -- example: ['Data Scientist', 'Data Science Associate', 'Data Science Consultant']
  "SalariesReported" REAL, -- example: [105.0, 95.0, 66.0]
  "Location" TEXT, -- example: ['Bangalore', 'Pune', 'Hyderabad']
  "Salary" TEXT -- example: ['₹6,48,573/yr', '₹11,91,950/yr', '₹8,36,874/yr']
)

CREATE TABLE "web_region" (
"id" INTEGER, -- example: [1, 2, 3]
  "name" TEXT -- example: ['Northeast', 'Midwest', 'Southeast']
)

CREATE TABLE "hardware_fact_gross_price" (
"product_code" TEXT, -- example: ['A0118150101', 'A0118150102', 'A0118150103']
  "fiscal_year" INTEGER, -- example: [2020, 2021]
  "gross_price" REAL -- example: [16.2323, 19.0573, 19.8577]
)

CREATE TABLE "hardware_fact_manufacturing_cost" (
"product_code" TEXT, -- example: ['A0118150101', 'A0118150102', 'A0118150103']
  "cost_year" INTEGER, -- example: [2020, 2021]
  "manufacturing_cost" REAL -- example: [5.0207, 5.5172, 5.718]
)

CREATE TABLE "university_course" (
"CourseNo" TEXT, -- example: ['FIN300', 'FIN450', 'FIN480']
  "CrsDesc" TEXT, -- example: ['FUNDAMENTALS OF FINANCE', 'PRINCIPLES OF INVESTMENTS', 'CORPORATE FINANCE']
  "CrsUnits" INTEGER -- example: [4]
)

CREATE TABLE "hardware_fact_sales_monthly" (
"date" TEXT, -- example: ['2019-09-01', '2019-10-01', '2019-11-01']
  "product_code" TEXT, -- example: ['A0118150101', 'A0118150102', 'A0118150103']
  "customer_code" INTEGER, -- example: [70002017, 70002018, 70003181]
  "sold_quantity" INTEGER, -- example: [137, 47, 57]
  "fiscal_year" INTEGER -- example: [2020, 2021]
)

