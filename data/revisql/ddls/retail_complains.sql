CREATE TABLE client (
    `client_id` text, -- client id. unique id client number.
    `sex` text, -- sex of client.
    `day` integer, -- day of the birthday.
    `month` integer, -- month of the birthday.
    `year` integer, -- year when is born.
    `age` integer, -- age. teenager: 13-19. adult: 19-65. elder: > 65.
    `social` text, -- social number. ssn: us id number for each person.
    `first` text, -- first name.
    `middle` text, -- middle name.
    `last` text, -- last name.
    `phone` text, -- phone number.
    `email` text, -- email. google email / account: @gamil.com. microsoft email / account: xxx@outlook.com.
    `address_1` text, -- address 1.
    `address_2` text, -- address 2. entire address = (address_1, address_2).
    `city` text, -- city.
    `state` text, -- state code.
    `zipcode` integer, -- zipcode.
    `district_id` integer, -- district id. district id number.
    primary key (client_id),
    foreign key (district_id) references district(district_id)
);

CREATE TABLE state (
    `StateCode` text, -- No description.
    `State` text, -- No description.
    `Region` text, -- No description.
    primary key (StateCode)
);

CREATE TABLE callcenterlogs (
    `Date received` date, -- complaint date.
    `Complaint ID` text, -- unique id number representing each complaint.
    `rand client` text, -- client id.
    `phonefinal` text, -- phone final. final phone number.
    `vru+line` text, -- voice response unit line.
    `call_id` integer, -- call id. id number identifying the call.
    `priority` integer, -- priority of the complaint. 0, 1, 2,. null: not available,. higher: -> higher priority,. -> more serious/urgent complaint.
    `type` text, -- type of complaint.
    `outcome` text, -- the outcome of processing of complaints.
    `server` text, -- server.
    `ser_start` text, -- server start. server start time. HH:MM:SS.
    `ser_exit` text, -- server exit. server exit time.
    `ser_time` text, -- server time. longer server time referring to more verbose/longer complaint.
    primary key ("Complaint ID"),
    foreign key ("rand client") references client(client_id)
);

CREATE TABLE reviews (
    `Date` date, -- No description.
    `Stars` integer, -- No description.
    `Reviews` text, -- No description.
    `Product` text, -- No description.
    `district_id` integer, -- No description.
    primary key ("Date"),
    foreign key (district_id) references district(district_id)
);

CREATE TABLE events (
    `Date received` date, -- Date received.
    `Product` text, -- complaining about which product.
    `Sub-product` text, -- sub product if exists. detailed product.
    `Issue` text, -- problems leading to this complaints.
    `Sub-issue` text, -- sub problems leading to this complaints if exists. more detailed issue.
    `Consumer complaint narrative` text, -- Consumer complaint narrative.
    `Tags` text, -- tags of client.
    `Consumer consent provided?` text, -- Tags Consumer consent provided?. whether the tags labeled under permission of the clients.  null, 'N/A' or empty value: indicating that the company didn't get the permission of consent.  if the value is not empty: customers provide the consent for this tag.
    `Submitted via` text, -- Submitted via.
    `Date sent to company` text, -- Date sent to company. delay of the complaints = 'Date sent to company'.
    `Company response to consumer` text, -- Company response to consumer.
    `Timely response?` text, -- whether the response of the company is timely.
    `Consumer disputed?` text, -- whether the consumer dispute about the response from the company.
    `Complaint ID` text, -- id number indicating which complaint.
    `Client_ID` text, -- Client ID. id number indicating which client.
    primary key ("Complaint ID", Client_ID),
    foreign key ("Complaint ID") references callcenterlogs("Complaint ID"),
    foreign key (Client_ID) references client(client_id)
);

CREATE TABLE district (
    `district_id` integer, -- district id. unique id number representing district.
    `city` text, -- city.
    `state_abbrev` text, -- state abbreviated code.
    `division` text, -- division.
    primary key (district_id),
    foreign key (state_abbrev) references state(StateCode)
);

