CREATE TABLE Community_Area (
    `community_area_no` integer, -- community area number. unique community area number.
    `community_area_name` text, -- community area name.
    `side` text, -- district.
    `population` text, -- population of the community.
    primary key (community_area_no)
);

CREATE TABLE Crime (
    `report_no` integer, -- report number. unique Numeric identifier of the crime report.
    `case_number` text, -- case number. Case number assigned to the reported crime. There is usually one case number for any reported crime except in the case of multiple homicide where a single report will produce one case number for each victim. if case_number appears > 1, it means this is a multiple homicide.
    `date` text, -- Date of the occurrence of the incident being reported.
    `block` text, -- A redacted address indicating in which city block the incident occurred.
    `iucr_no` text, -- Illinois Uniform Crime Reporting number. Illinois Uniform Crime Reporting code: a four digit code used to classify criminal incidents when taking reports.
    `location_description` text, -- location description. A description of the kind of area where incident occurred.
    `arrest` text, -- A true/false value indicating if an arrest has been made.
    `domestic` text, -- A true/false value indicating if the incident is as case of domestic violence.
    `beat` integer, -- No description.
    `district_no` integer, -- district number. A number indicating the geographic subdivision of the police district where the incident occurred.
    `ward_no` integer, -- ward number. A two digit number identifying the legislative district (ward) where the incident occurred.
    `community_area_no` integer, -- community area number. A number identifying one of 77 geographic areas where the incident occurred.
    `fbi_code_no` text, -- fbi code number. A code identifying the type of crime reported according to the classification scheme used by the FBI.
    `latitude` text, -- The latitude where incident is reported to have occurred.
    `longitude` text, -- The longitude where incident is reported to have occurred. The precise location / coordinate: combines the longitude and latitude for plotting purposes. (latitude, longitude).
    primary key (report_no),
    foreign key (ward_no) references Ward(ward_no),
    foreign key (iucr_no) references IUCR(iucr_no),
    foreign key (district_no) references District(district_no),
    foreign key (community_area_no) references Community_Area(community_area_no),
    foreign key (fbi_code_no) references FBI_Code(fbi_code_no)
);

CREATE TABLE FBI_Code (
    `fbi_code_no` text, -- fbi code number. unique code made by fbi to classify the crime cases.
    `title` text, -- Short description of the kind of crime.
    `description` text, -- Detailed description of the kind of crime.
    `crime_against` text, -- crime against. States who the crime is against. Values are Persons, Property, Society, or "Persons and Society".
    primary key (fbi_code_no)
);

CREATE TABLE Neighborhood (
    `neighborhood_name` text, -- neighborhood name. The neighborhood name as used in everyday communication.
    `community_area_no` integer, -- community area number. The community area in which most of the neighborhood is located.
    primary key (neighborhood_name),
    foreign key (community_area_no) references Community_Area(community_area_no)
);

CREATE TABLE Ward (
    `ward_no` integer, -- No description.
    `alderman_first_name` text, -- No description.
    `alderman_last_name` text, -- No description.
    `alderman_name_suffix` text, -- No description.
    `ward_office_address` text, -- No description.
    `ward_office_zip` text, -- No description.
    `ward_email` text, -- No description.
    `ward_office_phone` text, -- No description.
    `ward_office_fax` text, -- No description.
    `city_hall_office_room` integer, -- No description.
    `city_hall_office_phone` text, -- No description.
    `city_hall_office_fax` text, -- No description.
    `Population` text, -- No description.
    primary key (ward_no)
);

CREATE TABLE IUCR (
    `iucr_no` text, -- iucr number. Unique identifier for the incident classification.
    `primary_description` text, -- primary description. The general description of the incident classification. It's the general description.
    `secondary_description` text, -- secondary description. The specific description of the incident classification. It's the specific description.
    `index_code` text, -- index code. Uses values "I" and "N" to indicate crimes. â¢ "Indexed" (severe, such as murder, rape, arson, and robbery). â¢ "Non-Indexed" (less severe, such as vandalism, weapons violations, and peace disturbance).
    primary key (iucr_no)
);

CREATE TABLE District (
    `district_no` integer, -- district number. unique number identifying the district.
    `district_name` text, -- district name. name of the district.
    `address` text, -- Street address of the police district building.
    `zip_code` integer, -- zip code. ZIP code of the police district building.
    `commander` text, -- Name of the district's commanding officer. the person who should be responsible for the crime case.
    `email` text, -- Email address to contact district administrators.
    `phone` text, -- Main telephone number for the district.
    `fax` text, -- Fax number for the district.
    `tty` text, -- Number of the district teletype machine. A teletype machine is a device that lets people who are deaf, hard of hearing, or speech-impaired use the telephone to communicate, by allowing them to type text messages.
    `twitter` text, -- The district twitter handle. Twitter is a social networking service on which users post and interact with brief messages known as "tweets".
    primary key (district_no)
);

