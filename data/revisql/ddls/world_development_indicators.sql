CREATE TABLE SeriesNotes (
    `Seriescode` text, -- Series code. code identifying the series.
    `Year` text, -- year.
    `Description` text, -- Description of series.
    primary key (Seriescode, Year),
    foreign key (Seriescode) references Series(SeriesCode)
);

CREATE TABLE Indicators (
    `CountryName` text, -- Country code. code identifying unique countries.
    `CountryCode` text, -- Series code. Series code of countries.
    `IndicatorName` text, -- Indicator Name. indicator name.
    `IndicatorCode` text, -- Indicator Code. indicator code.
    `Year` integer, -- year.
    `Value` integer, -- value.
    primary key (CountryCode, IndicatorCode, Year),
    FOREIGN KEY (CountryCode) REFERENCES Country(CountryCode)
);

CREATE TABLE Footnotes (
    `Countrycode` text, -- Country code. code identifying unique countries.
    `Seriescode` text, -- Series code. Series code of countries.
    `Year` text, -- Year.
    `Description` text, -- Description of country footnotes.
    primary key (Countrycode, Seriescode, Year),
    FOREIGN KEY (Seriescode) REFERENCES Series(SeriesCode),
    FOREIGN KEY (Countrycode) REFERENCES Country(CountryCode)
);

CREATE TABLE CountryNotes (
    `Countrycode` text, -- Country code. code identifying unique countries.
    `Seriescode` text, -- Series code. Series code of countries.
    `Description` text, -- description.
    primary key (Countrycode, Seriescode),
    FOREIGN KEY (Seriescode) REFERENCES Series(SeriesCode),
    FOREIGN KEY (Countrycode) REFERENCES Country(CountryCode)
);

CREATE TABLE Series (
    `SeriesCode` text, -- Series Code. unique code identifying series.
    `Topic` text, -- topic of series.
    `IndicatorName` text, -- Indicator Name.
    `ShortDefinition` text, -- Short Definition. Short Definition of series.
    `LongDefinition` text, -- Long Definition. Long Definition of series.
    `UnitOfMeasure` text, -- Unit Of Measure.
    `Periodicity` text, -- No description.
    `BasePeriod` text, -- Base Period. a period of business or economic activity used as a basis or reference point especially for indexing, calculating, estimating, or adjudicating prices, taxes, compensation, income, and production.
    `OtherNotes` integer, -- Other Notes.
    `AggregationMethod` text, -- Aggregation Method.
    `LimitationsAndExceptions` text, -- Limitations And Exceptions.
    `NotesFromOriginalSource` text, -- Notes From Original Source.
    `GeneralComments` text, -- General Comments.
    `Source` text, -- source of this data.
    `StatisticalConceptAndMethodology` text, -- Statistical Concept And Methodology.
    `DevelopmentRelevance` text, -- Development Relevance.
    `RelatedSourceLinks` text, -- Related Source Links.
    `OtherWebLinks` integer, -- Other Web Links.
    `RelatedIndicators` integer, -- Related Indicators.
    `LicenseType` text, -- License Type.
    primary key (SeriesCode)
);

CREATE TABLE Country (
    `CountryCode` text, -- Country Code. unique code identifying countries.
    `ShortName` text, -- Short Name. Short names of countries.
    `TableName` text, -- Table Name. table names of countries.
    `LongName` text, -- Long Name. long or full name of countries.
    `Alpha2Code` text, -- Alpha to Code. 2 digit code of countries.
    `CurrencyUnit` text, -- Currency Unit. Currency Unit used in this country.
    `SpecialNotes` text, -- Special Notes.
    `Region` text, -- region that country belongs to.
    `IncomeGroup` text, -- Income Group. income level of countries.
    `Wb2Code` text, -- world bank to code.
    `NationalAccountsBaseYear` text, -- National Accounts Base Year. the year used as the base period for constant price calculations in the country's national accounts.
    `NationalAccountsReferenceYear` text, -- National Accounts Reference Year.
    `SnaPriceValuation` text, -- SNA Price Valuation.
    `LendingCategory` text, -- Lending Category. Lending category. • IDA: International Development Associations: (IDA) is the part of the World Bank that helps the world's poorest countries. • IBRD: The International Bank for Reconstruction and Development (IBRD) is a global development cooperative owned by 189 member countries. • Blend: Blend is the cloud banking infrastructure powering billions of dollars in financial transactions every day.
    `OtherGroups` text, -- Other groups. other groups. • HIPC: Heavily Indebted Poor Countries. • Euro Area: The euro area consists of those Member States of the European Union that have adopted the euro as their currency.
    `SystemOfNationalAccounts` text, -- System Of National Accounts.
    `AlternativeConversionFactor` text, -- Alternative Conversion Factor. Alternative conversion factor is the underlying annual exchange rate used for the World Bank Atlas method.
    `PppSurveyYear` text, -- purchasing power parity survey year.
    `BalanceOfPaymentsManualInUse` text, -- Balance Of Payments Manual In Use.
    `ExternalDebtReportingStatus` text, -- External Debt Reporting Status. • Actual. • Preliminary. • Estimate. commonsense reasoning:. If ExternalDebtReportingStatus='Actual', it means this external debt reporting is real and actual, and finished. if 'Estimate', it means external debt reporting is finished by estimation. if 'preliminary', it means this external debt reporting is not finished.
    `SystemOfTrade` text, -- System Of Trade.
    `GovernmentAccountingConcept` text, -- Government Accounting Concept.
    `ImfDataDisseminationStandard` text, -- International Monetory Fund Data Dissemination Standard. IMF Standards for Data Dissemination.
    `LatestPopulationCensus` text, -- Latest Population Census.
    `LatestHouseholdSurvey` text, -- Latest Household Survey.
    `SourceOfMostRecentIncomeAndExpenditureData` text, -- Source Of Most Recent Income And Expenditure Data.
    `VitalRegistrationComplete` text, -- Vital Registration Complete.
    `LatestAgriculturalCensus` text, -- Latest Agricultural Census.
    `LatestIndustrialData` integer, -- Latest Industrial Data.
    `LatestTradeData` integer, -- Latest Trade Data.
    `LatestWaterWithdrawalData` integer, -- Latest Water Withdrawal Data.
    primary key (CountryCode)
);

