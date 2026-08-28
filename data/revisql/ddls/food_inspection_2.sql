CREATE TABLE inspection (
    `inspection_id` integer, -- inspection id. the unique id for the inspection.
    `inspection_date` date, -- inspection date. the date of the inspection. yyyy-mm-dd.
    `inspection_type` text  s, -- inspection type. the type of the inspection.
    `results` text, -- the inspection results.  Pass.  Pass w/ Conditions.  Fail.  Out of Business.  Business Not Located.  No Entry.  Not Ready. the quality is verified: Pass + Pass w/ Conditions.
    `employee_id` integer, -- employee id. the id of the employee who is responsible for the inspection.
    `license_no` integer, -- license number. the license number of the facility that is inspected.
    `followup_to` integer, -- followup to. the id of inspections that this inspection is following up. 'No data' means the inspection has no followup. if followup_to has value:.  it's the parent inspection id, and this value represents child inspection id.  the entire inspection: followup_to + inspection_id.
    primary key (inspection_id),
    foreign key (employee_id) references employee(employee_id),
    foreign key (license_no) references establishment(license_no),
    foreign key (followup_to) references inspection(inspection_id)
);

CREATE TABLE establishment (
    `license_no` integer, -- license number. the license number of the facility.
    `dba_name` text, -- doing business as name. the dba name of the facility. DBA stands for doing business as. It's also referred to as your business's assumed, trade or fictitious name.
    `aka_name` text, -- as know as name. the aka name of the facility. the aka name is used to indicate another name that a person or thing has or uses.
    `facility_type` text, -- facility type. the facility type. can ask some questions about its categories, for i.e.:.  "Restaurant" and "Cafeteria" refers to restaurants.  PASTRY school and "CULINARY ARTS SCHOOL" belongs to "Education" or "School". Please design this and mention your evidence in the commonsense part of questions, thanks.
    `risk_level` integer, -- risk level. the risk level of the facility. Facilities with higher risk level have more serious food safety issues.
    `address` text, -- physical street address of the facility.
    `city` text, -- city of the facility's address.
    `state` text, -- state of the facility's address.
    `zip` integer, -- postal code of the facility's address.
    `latitude` real, -- the latitude of the facility.
    `longitude` real, -- the longitude of the facility. location = POINT(latitude, longitude).
    `ward` integer, -- the ward number of the facility. Ward number is the number assigned by the local authorities to identify the population count in each town, village and district usually for electoral purposes or to extract property and locality details.
    primary key (license_no)
);

CREATE TABLE employee (
    `employee_id` integer, -- employee id. the unique id for the employee.
    `first_name` text, -- first name. first name of the employee.
    `last_name` text, -- last name. last name of the employee. full name: first name + last_name.
    `address` text, -- physical street address of the employee.
    `city` text, -- city of the employee's address.
    `state` text, -- state of the customer's address.
    `zip` integer, -- postal code of the employee's address.
    `phone` text, -- telephone number to reach the customer.
    `title` text, -- career title of the employee.  Sanitarian.  Supervisor.  Division Manager.
    `salary` integer, -- the salary of the employee. us dollars / year. monthly salary: salary / 12.
    `supervisor` integer, -- the employee id of the employee's supervisor.
    primary key (employee_id),
    foreign key (supervisor) references employee(employee_id)
);

CREATE TABLE inspection_point (
    `point_id` integer, -- point id. the unique id for the inspection point.
    `Description` text, -- the specific description of the inspection results.
    `category` text, -- the inspection category.
    `code` text, -- the sanitary operating requirement code. Each code stands for an operating requirement.
    `fine` integer, -- the fines for food safety problems. The fine of more serious food safety problems will be higher.  Minor: 100.  Serious: 250.  Critical: 500.
    `point_level` text, -- point level. Critical / Serious/ Minor.
    primary key (point_id)
);

CREATE TABLE violation (
    `inspection_id` integer, -- inspection id. the unique id for the inspection.
    `point_id` integer, -- point id. the inspection point id.
    `fine` integer, -- the fine of the violation. The fine of more serious food safety problems will be higher.  Minor: 100.  Serious: 250.  Critical: 500.
    `inspector_comment` text, -- the comment of the inspector.
    primary key (inspection_id, point_id),
    foreign key (inspection_id) references inspection(inspection_id),
    foreign key (point_id) references inspection_point(point_id)
);

